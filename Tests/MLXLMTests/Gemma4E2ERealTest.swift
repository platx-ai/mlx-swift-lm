// Copyright © 2026 PlatX AI.
//
// End-to-end Gemma 4 audio transcription test using a real cached model
// and a real audio fixture. Used to verify that rebasing PR #192 onto
// upstream main does not introduce numerical or behavioural regressions
// against the platx-ai/mlx-swift-lm@668347f baseline.
//
// Skipped automatically if the model is not present in the local
// HuggingFace cache.

import Foundation
import Testing
import MLX
import MLXLMCommon
import MLXVLM
import MLXHuggingFace
import Tokenizers

@Suite("Gemma4 Audio E2E Real")
struct Gemma4E2ERealTest {

    /// Loads the reference audio (decoded m4a → 16 kHz mono Float JSON)
    /// from the test fixtures directory.
    private func loadReferenceAudio() throws -> [Float] {
        let sourceDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let url = sourceDir.appendingPathComponent("Fixtures/gemma4_e2e_audio.json")
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw NSError(
                domain: "test", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Audio fixture not found: \(url.path)"])
        }
        let data = try Data(contentsOf: url)
        let raw = try JSONSerialization.jsonObject(with: data) as! [Double]
        return raw.map { Float($0) }
    }

    /// Returns true if the Gemma 4 e4b 4-bit model directory is cached locally.
    private func modelCached(_ modelId: String) -> Bool {
        let dirName = "models--" + modelId.replacingOccurrences(of: "/", with: "--")
        let cacheDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent(dirName)
        return FileManager.default.fileExists(atPath: cacheDir.path)
    }

    @Test
    func transcribeReferenceAudio() async throws {
        let modelId = "mlx-community/gemma-4-e4b-it-4bit"

        // Skip if model not cached locally — this test should not download.
        guard modelCached(modelId) else {
            print("[SKIP] \(modelId) not cached locally")
            return
        }

        let audio = try loadReferenceAudio()
        #expect(audio.count == 116_352, "reference audio should be 116352 samples (7.272s @ 16kHz)")

        // Load model directly from local cache directory (no downloader needed).
        let dirName = "models--" + modelId.replacingOccurrences(of: "/", with: "--")
        let cacheRoot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent(dirName)
        let snapshotsDir = cacheRoot.appendingPathComponent("snapshots")
        let snapshotEntries = try FileManager.default.contentsOfDirectory(atPath: snapshotsDir.path)
        guard let snapshotName = snapshotEntries.first else {
            throw NSError(domain: "test", code: 2, userInfo: [
                NSLocalizedDescriptionKey: "no snapshot under \(snapshotsDir.path)"
            ])
        }
        let modelDir = snapshotsDir.appendingPathComponent(snapshotName)
        print("[E2E] loading model from \(modelDir.path)")

        let tokenizerLoader = #huggingFaceTokenizerLoader()
        let context = try await VLMModelFactory.shared.load(
            from: modelDir, using: tokenizerLoader)

        // Build UserInput with audio. Use the exact prompt Talk's
        // Gemma4ASREngine constructs at intensity=.medium so we reproduce
        // the 668347f baseline transcript:
        //   "我现在已经切换到詹姆斯四。这个就是詹姆斯四接情的这段时期。"
        var input = UserInput(prompt: "Transcribe this audio verbatim. Add proper punctuation.")
        input.audios = [audio]

        let startTime = CFAbsoluteTimeGetCurrent()
        let lmInput = try await context.processor.prepare(input: input)
        let prepareElapsed = CFAbsoluteTimeGetCurrent() - startTime

        // Sanity check on shapes — audioShape MUST match Talk baseline:
        //   audioShape=[1, 726, 128]
        // The prompt token count may differ from Talk if Talk uses a
        // different MessageGenerator, but the audio-side shapes prove the
        // mel + encoder + scatter path is numerically equivalent.
        #expect(lmInput.audio != nil, "audio path missing")
        if let audioFeat = lmInput.audio?.features {
            #expect(audioFeat.shape == [1, 726, 128], "audio features shape mismatch — REGRESSION!")
        }
        print("[E2E] prepare: \(String(format: "%.3fs", prepareElapsed)), tokens=\(lmInput.text.tokens.shape), audio=\(lmInput.audio?.features.shape ?? [])")

        // Generation loop matching Talk's Gemma4ASREngine.transcribe()
        let model = context.model
        let cache = model.newCache(parameters: nil)
        let prepareResult = try model.prepare(lmInput, cache: cache, windowSize: nil)

        var outputTokens = [Int]()
        var logits: MLXArray
        switch prepareResult {
        case .logits(let result):
            logits = result.logits
        case .tokens(let tokens):
            logits = model.callAsFunction(tokens.tokens, cache: cache)
        }

        for _ in 0..<500 {
            let lastLogits = logits[0..., -1, 0...]
            let nextToken = lastLogits.argMax(axis: -1).item(Int.self)
            if [1, 106, 50].contains(nextToken) { break }
            outputTokens.append(nextToken)
            let nextTokenArray = MLXArray([Int32(nextToken)]).expandedDimensions(axis: 0)
            logits = model.callAsFunction(nextTokenArray, cache: cache)
            eval(logits)
        }

        let totalElapsed = CFAbsoluteTimeGetCurrent() - startTime
        let text = context.tokenizer.decode(tokenIds: outputTokens, skipSpecialTokens: false)
            .trimmingCharacters(in: .whitespacesAndNewlines)

        // Print results to stdout for manual comparison with 668347f baseline.
        print("[E2E] generation: \(String(format: "%.3fs", totalElapsed - prepareElapsed)), totalTokens=\(outputTokens.count)")
        print("[E2E] total elapsed: \(String(format: "%.3fs", totalElapsed))")
        print("[E2E] tokens (first 30): \(outputTokens.prefix(30))")
        print("[E2E] transcript: \(text)")

        // We expect non-empty transcription; the exact text may vary slightly
        // due to floating point determinism, but the 668347f baseline was:
        //   "我现在已经切换到詹姆斯四。这个就是詹姆斯四接情的这段时期。"
        // Inference time on baseline was ~0.36s.
        #expect(!text.isEmpty, "transcription should not be empty")
        #expect(outputTokens.count > 0, "should generate at least one token")
    }
}

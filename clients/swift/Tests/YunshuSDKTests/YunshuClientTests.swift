import Testing
import Foundation
@testable import YunshuSDK

@Suite("YunshuClient Tests")
struct YunshuClientTests {

    @Test("Client initializes with trailing-slash trimming")
    func initTrimsSlash() {
        let client = YunshuClient(baseURL: "http://localhost:8000/")
        #expect(client.baseURL.absoluteString == "http://localhost:8000")
    }

    @Test("Client stores optional API key")
    func apiKeyStored() {
        let a = YunshuClient(baseURL: "http://localhost:8000")
        #expect(a.apiKey == nil)

        let b = YunshuClient(baseURL: "http://localhost:8000", apiKey: "sk-test")
        #expect(b.apiKey == "sk-test")
    }

    @Test("YunshuError has readable descriptions")
    func errorDescriptions() {
        let cases: [YunshuError] = [
            .networkError("timeout"),
            .apiError("model not found"),
            .decodingError,
            .authError,
        ]
        for e in cases {
            #expect(e.errorDescription != nil)
        }
    }

    @Test("ChatResponse decodes from JSON")
    func decodeChatResponse() throws {
        let json = """
        {
          "id": "chatcmpl-123",
          "object": "chat.completion",
          "created": 1700000000,
          "model": "qwen2.5-0.5b-instruct",
          "choices": [
            {
              "index": 0,
              "message": {"role": "assistant", "content": "Hello!"},
              "finish_reason": "stop"
            }
          ],
          "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 2,
            "total_tokens": 7
          }
        }
        """.data(using: .utf8)!

        let resp = try JSONDecoder().decode(ChatResponse.self, from: json)
        #expect(resp.id == "chatcmpl-123")
        #expect(resp.choices.count == 1)
        #expect(resp.choices[0].message.content == "Hello!")
        #expect(resp.choices[0].finishReason == "stop")
        #expect(resp.usage?.totalTokens == 7)
    }

    @Test("CompletionResponse decodes from JSON")
    func decodeCompletionResponse() throws {
        let json = """
        {
          "id": "cmpl-456",
          "object": "text_completion",
          "created": 1700000000,
          "model": "qwen2.5-0.5b-instruct",
          "choices": [
            {
              "index": 0,
              "text": " world",
              "finish_reason": "length"
            }
          ]
        }
        """.data(using: .utf8)!

        let resp = try JSONDecoder().decode(CompletionResponse.self, from: json)
        #expect(resp.choices[0].text == " world")
        #expect(resp.choices[0].finishReason == "length")
    }

    @Test("EmbeddingResponse decodes from JSON")
    func decodeEmbeddingResponse() throws {
        let json = """
        {
          "object": "list",
          "data": [
            {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}
          ],
          "model": "text-embedding-small",
          "usage": {"prompt_tokens": 3, "completion_tokens": 0, "total_tokens": 3}
        }
        """.data(using: .utf8)!

        let resp = try JSONDecoder().decode(EmbeddingResponse.self, from: json)
        #expect(resp.data.count == 1)
        #expect(resp.data[0].embedding == [0.1, 0.2, 0.3])
    }

    @Test("ModelInfo decodes from JSON")
    func decodeModelInfo() throws {
        let json = """
        {
          "id": "qwen2.5-0.5b-instruct",
          "object": "model",
          "created": 1700000000,
          "owned_by": "yunshu"
        }
        """.data(using: .utf8)!

        let info = try JSONDecoder().decode(ModelInfo.self, from: json)
        #expect(info.id == "qwen2.5-0.5b-instruct")
        #expect(info.ownedBy == "yunshu")
    }
}

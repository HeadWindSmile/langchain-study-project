package com.example.ragspring.client;

import com.example.ragspring.model.ApiResponse;
import com.example.ragspring.model.AskData;
import com.example.ragspring.model.AskRequest;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

@Component
public class RagClient {

    private final RestClient ragRestClient;

    public RagClient(RestClient ragRestClient) {
        this.ragRestClient = ragRestClient;
    }

    public ApiResponse<AskData> ask(String question) {
        return ragRestClient.post()
                .uri("/ask")
                .contentType(MediaType.APPLICATION_JSON)
                .body(new AskRequest(question))
                .retrieve()
                .body(new ParameterizedTypeReference<ApiResponse<AskData>>() {});
    }
}
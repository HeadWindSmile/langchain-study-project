package com.example.ragspring.client;

import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;

import java.util.Map;

@Component
public class RagStreamClient {

    private final WebClient webClient;

    public RagStreamClient() {
        this.webClient = WebClient.builder()
                .baseUrl("http://127.0.0.1:8000")
                .defaultHeader("X-API-Key", "dev-secret-key")
                .build();
    }

    public Flux<ServerSentEvent<String>> streamAsk(String question) {
        return webClient.post()
                .uri("/ask/stream")
                .contentType(MediaType.APPLICATION_JSON)
                .accept(MediaType.TEXT_EVENT_STREAM)
                .bodyValue(Map.of("question", question))
                .retrieve()
                .bodyToFlux(new ParameterizedTypeReference<ServerSentEvent<String>>() {});
    }
}

package com.example.ragspring.controller;

import com.example.ragspring.client.RagStreamClient;
import com.example.ragspring.model.AskRequest;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

@RestController
@RequestMapping("/api/chat")
public class ChatStreamController {

    private final RagStreamClient ragStreamClient;

    public ChatStreamController(RagStreamClient ragStreamClient) {
        this.ragStreamClient = ragStreamClient;
    }

    //@PostMapping(value = "/stream", produces = "text/event-stream;charset=UTF-8")
    //public Flux<ServerSentEvent<String>> stream(@RequestBody AskRequest request) {
    //    return ragStreamClient.streamAsk(request.question());
    //}

    @PostMapping("/stream")
    public ResponseEntity<Flux<ServerSentEvent<String>>> stream(@RequestBody AskRequest request) {
        MediaType sseUtf8 = new MediaType("text", "event-stream", java.nio.charset.StandardCharsets.UTF_8);
        return ResponseEntity.ok()
                .contentType(sseUtf8)
                .body(ragStreamClient.streamAsk(request.question()));
    }

}
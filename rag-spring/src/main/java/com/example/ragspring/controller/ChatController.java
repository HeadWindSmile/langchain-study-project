package com.example.ragspring.controller;

import com.example.ragspring.client.RagClient;
import com.example.ragspring.model.ApiResponse;
import com.example.ragspring.model.AskData;
import com.example.ragspring.model.AskRequest;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private final RagClient ragClient;

    public ChatController(RagClient ragClient) {
        this.ragClient = ragClient;
    }

    @PostMapping
    public ApiResponse<AskData> chat(@RequestBody AskRequest request) {
        System.out.println("Java接收到的前端请求: " + request);
        return ragClient.ask(request.question());
    }
}
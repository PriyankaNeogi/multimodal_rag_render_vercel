"use client";

import { useState } from "react";
import { BACKEND_URL } from "@/lib/backend";

export default function ChatUI() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!message.trim()) return;

    setLoading(true);
    setResponse("");

    try {
      const res = await fetch(`${BACKEND_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ query: message }),
      });

      const data = await res.json();
      setResponse(data.answer || "No response received");
    } catch (error) {
      console.error(error);
      setResponse("Error connecting to backend");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Multimodal RAG Chatbot</h1>

      <textarea
        className="w-full border p-2 rounded mb-2"
        rows={4}
        placeholder="Ask something..."
        value={message}
        onChange={(e) => setMessage(e.target.value)}
      />

      <button
        onClick={sendMessage}
        disabled={loading}
        className="bg-black text-white px-4 py-2 rounded"
      >
        {loading ? "Thinking..." : "Send"}
      </button>

      {response && (
        <div className="mt-4 p-3 border rounded bg-gray-100">
          <strong>Response:</strong>
          <p>{response}</p>
        </div>
      )}
    </div>
  );
}

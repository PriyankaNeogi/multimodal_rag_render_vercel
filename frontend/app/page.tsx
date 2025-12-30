"use client";

import { useState } from "react";

export default function Page() {
  const [msg, setMsg] = useState("");
  const [ans, setAns] = useState("");

  async function ask() {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: msg })
    });

    const data = await res.json();
    setAns(data.answer);
  }

  return (
    <div style={{ padding: 20 }}>
      <h2>Multimodal RAG Chat</h2>

      <textarea
        rows={4}
        style={{ width: "100%" }}
        onChange={e => setMsg(e.target.value)}
      />

      <br />
      <button onClick={ask}>Ask</button>

      <p>{ans}</p>
    </div>
  );
}

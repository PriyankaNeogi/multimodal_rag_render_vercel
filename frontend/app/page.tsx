"use client";

import { useState } from "react";

export default function Page() {
  const [msg, setMsg] = useState("");
  const [ans, setAns] = useState("");
  const [file, setFile] = useState<File | null>(null);

  async function uploadPdf() {
    if (!file) return;

    const form = new FormData();
    form.append("file", file);

    await fetch("/api/upload", {
      method: "POST",
      body: form,
    });

    alert("PDF uploaded");
  }

  async function ask() {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: msg }),
    });

    const data = await res.json();
    setAns(data.answer);
  }

  return (
    <div style={{ padding: 20 }}>
      <h2>Multimodal RAG Chat</h2>

      <input
        type="file"
        accept="application/pdf"
        onChange={(e) => setFile(e.target.files?.[0] || null)}
      />
      <button onClick={uploadPdf}>Upload PDF</button>

      <br /><br />

      <textarea
        rows={4}
        style={{ width: "100%" }}
        onChange={(e) => setMsg(e.target.value)}
      />

      <button onClick={ask}>Ask</button>

      <p>{ans}</p>
    </div>
  );
}

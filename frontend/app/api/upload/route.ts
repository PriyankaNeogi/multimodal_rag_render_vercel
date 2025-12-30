import { NextResponse } from "next/server";

const BACKEND_URL =
  "https://multimodal-rag-backend-g1i2.onrender.com";

export async function POST(req: Request) {
  const formData = await req.formData();

  const res = await fetch(`${BACKEND_URL}/upload-pdf`, {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  return NextResponse.json(data);
}

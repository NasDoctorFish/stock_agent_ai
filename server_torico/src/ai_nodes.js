import OpenAI from "openai";
import dotenv from "dotenv";
import fetch, { Headers, Request, Response } from "node-fetch";

globalThis.fetch = fetch;
globalThis.Headers = Headers;
globalThis.Request = Request;
globalThis.Response = Response;


dotenv.config();

const client = new OpenAI({
  apiKey: process.env.CHATGPT_API, // 네가 쓰는 env 키 이름 그대로
});

export async function askChatGPT(userText) {
  const resp = await client.responses.create({
    model: "gpt-5.2",
    input: userText,
  });

  // 응답 텍스트(편의 필드)
  return resp.output_text;
}

//
async function run() {
  const prompt = process.argv.slice(2).join(" ");

  const resp = await client.responses.create({
    model: "gpt-5.2",
    input: prompt
  });

  console.log("==== GPT OUTPUT ====");
  console.log(resp.output_text);
}

run();

import "dotenv/config"
import OpenAI from "openai"
import readline from "readline/promises"

const client = new OpenAI({
    apiKey: process.env.DASHSCOPE_API_KEY,
    baseURL: "https://ws-op1rmhcapcp4k4fg.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
});


const tools: OpenAI.Responses.Tool[] = [
    { type: "web_search"},
    { type: "code_interpreter"},
    { type: "web_extractor"}
]

async function getResponse(conversation: OpenAI.Responses.ResponseInput) {
    const response = await client.responses.create({
        model: "qwen3.7-plus",
        input: conversation,
        tools: tools,
        instructions: "语气像人与人之间的日常对话, 内容无需详细和结构化",
    });
    return response;
}

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
})

const conversation: OpenAI.Responses.ResponseInput = [
    { role: "system", content: "You are my friend." },
];

async function main() {
    for (;;) {
        const input = await rl.question("> ");
        if (input === "exit") {
            rl.close();
            break;
        }

        conversation.push({ role: "user", content: input });

        const response = await getResponse(conversation);
        console.log(response.output_text);

        conversation.push({ role: "assistant", content: response.output_text });
    }
}

main();

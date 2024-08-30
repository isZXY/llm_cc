import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer,AutoModelForCausalLM,AutoTokenizer


model_dir = "../Meta-Llama-3-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_dir)

tokenizer = AutoTokenizer.from_pretrained(model_dir)


prompt_ = (
    f"<|Task description|>You are tasked with selecting the most appropriate congestion control algorithm based on the provided network statistics and scenario. Your goal is to either switch to or maintain the algorithm that maximizes Quality of Experience (QoE) from the given options.\n"
    f"<|Given algos introduction|>BBR: Optimized for low-latency scenarios. Cubic: Best suited for high-throughput needs. Reno: A classic loss-based algorithm.\n"
    f"<|Network stat|>Bandwidth: max 12 Mbps, min 9 Mbps; RTT: max 500 ms, min 300 ms; Packet loss: max 0.05%, min 0.01%\n"
    f"<|Scenario|>Cloud gaming\n"
    f"<|Current algorithm|>BBR\n"
    f"Example:<|Network stat|>Bandwidth: max 4 Mbps, min 1 Mbps; RTT: max 72 ms, min 42 ms; Packet loss: max 0.5%, min 0.09% <|Scenario|>file download<|Current algorithm|>BBR <|Output|>Cubic\n"
    f"<|Output|>_______(Algo Name)"
)


for i in range(6):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device=1
    )

    sequences = pipeline(
        prompt_,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=1200,
    )


    for seq in sequences:
        print(f"Result: {seq['generated_text']}")

    print("------------------------------------------------")
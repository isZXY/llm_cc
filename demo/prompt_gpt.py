from openai import OpenAI

# 设置中转API地址
api_base = "https://api.chatanywhere.tech/v1"
api_key = "sk-CelumY6pSozc9ZVHQSbjdVNk10LxRONiIBo5JxgRUxHShq5z"

prompt_ = (
    f"<|Given algos introduction|>BBR: Optimized for low-latency scenarios. Cubic: Best suited for high-throughput needs. Reno: A classic loss-based algorithm.\n"
    f"<|Network stat|>Bandwidth: max 12 Mbps, min 9 Mbps; RTT: max 500 ms, min 300 ms; Packet loss: max 0.05%, min 0.01%\n"
    f"<|Scenario|>Cloud gaming\n"
    f"<|Current algorithm|>BBR\n"
    f"Example:<|Network stat|>Bandwidth: max 4 Mbps, min 1 Mbps; RTT: max 72 ms, min 42 ms; Packet loss: max 0.5%, min 0.09% <|Scenario|>file download<|Current algorithm|>BBR <|Output|>Cubic\n"
    f"<|Output|> _______ (Algo Name),Please only give the algo name."
)

client = OpenAI(
    api_key=api_key,
    base_url=api_base
)


for i in range(5):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are tasked with selecting the most appropriate congestion control algorithm based on the provided network statistics and scenario. Your goal is to either switch to or maintain the algorithm that maximizes Quality of Experience (QoE) from the given options."},
            {
                "role": "user",
                "content": prompt_
            }
        ],
        logprobs=True,
    )

    print(completion.choices[0].message)

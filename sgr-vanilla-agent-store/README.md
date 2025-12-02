# Sample Agent for STORE benchmark at ERC platform

This agent demonstrates how to build a simple chatbot capable of automating processes in an online store. It is not designed to be state of the art, but rather be something readable and compact, to show how to get started.


Check out [STORE benchmark](https://erc.timetoact-group.at/benchmarks/store) for the leaderboard and more information about the benchmark. Check out [SDK README.md](../README.MD) for more details about this project and SDK

This agent doesn't use any external libraries aside from OpenAI SDK and ERC3 SDK. Files:

- [requirements.txt](requirements.txt) - dependencies.
- [main.py](main.py) - entry point that connects to the ERC platform and gets a list of tasks
- [store_agent.py](store_agent.py) - agent itself. It uses [Schema-Guided Reasoning](https://abdullin.com/schema-guided-reasoning/) and is based on simple [SGR NextStep architecture](https://abdullin.com/schema-guided-reasoning/demo)

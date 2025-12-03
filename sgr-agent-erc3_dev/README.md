# Sample Agent for ERC3 benchmark at ERC platform

This agent demonstrates how to build a simple chatbot capable of automating enterprise APIs in a complex company. It is not designed to be state of the art, but rather something readable and compact.

Read [Project README.md](../README.MD) for more details about this repository. Benchmarks and their leaderboards:

- [ERC3-DEV](https://erc.timetoact-group.at/benchmarks/erc3-dev) - get started with this one
- [ERC3-TEST](https://erc.timetoact-group.at/benchmarks/erc3-test) - more complex, includes subtle changes in companies
- ERC3-PROD - Coming soon, December 9th!

This agent doesn't use any external libraries aside from OpenAI SDK and ERC3 SDK. Files:

- [requirements.txt](requirements.txt) - dependencies.
- [main.py](main.py) - entry point that connects to the ERC platform and gets a list of tasks
- [agent.py](agent.py) - agent itself. It uses [Schema-Guided Reasoning](https://abdullin.com/schema-guided-reasoning/) and is based on simple [SGR NextStep architecture](https://abdullin.com/schema-guided-reasoning/demo)

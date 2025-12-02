# ERC3 Agents

Это репозиторий с агентами под два бенчмарка ERC3.

- **store** — Benchmark for an online shop with a product catalogue, discounts and checkout basket. Agent needs to purchase proper products by putting them into the basket and checking out. Terminate task early, if it is not doable.
- **erc3-dev** — Benchmark with a set of APIs for the Enterprise RAG Challenge 3: AI Agents. This is Aetherion Analytics Gmb. Check out company wiki via API for more insights. Especially check the rulebook.md. NB: In production ERC3, there will be multiple companies with multiple different backstories.

## Что лежит в репо
- `sgr-vanilla-agent-store` — базовый Schema-Guided Reasoning агент для магазина на OpenAI (`gpt-4.1-mini` по умолчанию). (копия vanilla-агента из https://github.com/trustbit/erc3-agents)
- `sgr-knowledge-agent-store` — вариант для магазина с RAG на векторке, поверх smolagents; можно переключить OpenAI или Nebius (`gpt-5.1`).
- `sgr-vanilla-agent-erc3` — базовый Schema-Guided Reasoning агент для erc3-dev (копия vanilla-агента из https://github.com/trustbit/erc3-agents/tree/main/sgr-agent-erc3).
- `res` — скриншоты и вспомогательные артефакты.

## Требования и переменные окружения
- Python 3.10+ и `pip`.
- Ключ ERC3: `ERC3_API_KEY`.
- Ключи LLM: `OPENAI_API_KEY` (для OpenAI) и, при работе через Nebius, `NEBIUS_API_KEY` + опционально `NEBIUS_API_BASE`.

Пример `.env`:
```
ERC3_API_KEY=key-...
OPENAI_API_KEY=sk-...
NEBIUS_API_KEY=nb-...
# NEBIUS_API_BASE=https://api.studio.nebius.com/v1/

```

## Быстрый старт
1) Создайте и активируйте виртуальное окружение.  
2) Скопируйте `.env.example` или заполните `.env` по образцу выше.  
3) Установите зависимости и запустите нужный агент.

### Магазин (vanilla)
```
cd sgr-vanilla-agent-store
pip install -r requirements.txt
python main.py
```
Параметры модели, workspace и имени агента меняются в `sgr-vanilla-agent-store/main.py`.

### Магазин (knowledge)
```
cd sgr-knowledge-agent-store
pip install -r requirements.txt
python main.py
```
Для переключения провайдера обновите `provider` в `run_agent` и убедитесь, что заданы соответствующие ключи.

### ERC3-dev (NextStep SGR)
```
cd sgr-vanilla-agent-erc3
pip install -r requirements.txt
python main.py
```
Изучите `sgr-vanilla-agent-erc3/main.py` и `store_agent.py` для настройки модели, workspace и поведения.

## Полезное
- Агент для магазина работает с ERC3 SDK и отправляет сессии через `submit_session`.
- UI бенчмарка магазина можно посмотреть на https://erc.timetoact-group.at/ (см. скрин `res/console.png`).

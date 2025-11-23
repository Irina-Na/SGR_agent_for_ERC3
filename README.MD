# ERC3 Store Agents

Два готовых агента для бенчмарка ERC3 "store": простой SGR‑агент (NextStep от https://github.com/trustbit/erc3-agents) и версия с памятью, экстракцией знаний и агентом кодировщиком от smolagents.

## Что внутри
- `sgr-vanilla-agent-store` — минимальный Schema-Guided Reasoning цикл с OpenAI (`gpt-4.1-mini` по умолчанию).
- `sgr-knowledge-agent-store` — расширенный агент: формулирует критерии успеха, ведет сжатую память, умеет прогонять код через smolagents, работает как с OpenAI, так и с Nebius (`gpt-5.1` по умолчанию).
- `/res` — консольные скриншоты результата.

## Требования
- Python 3.10+ и `pip`.
- Аккаунт в ERC3 и `ERC3_API_KEY`.
- Ключ для LLM: `OPENAI_API_KEY` (для обоих), опционально `NEBIUS_API_KEY` и `NEBIUS_API_BASE` (для Nebius, базовый URL уже зашит).

Пример `.env` в корне:
```
ERC3_API_KEY=key-...
OPENAI_API_KEY=sk-...
NEBIUS_API_KEY=nb-...          # нужно только knowledge-агенту
# NEBIUS_API_BASE=https://api.studio.nebius.com/v1/  # если хотите переопределить
```

## Быстрый старт
1) Клонируйте репозиторий и при желании создайте виртуальное окружение.  
2) Положите `.env` в корень (ключи подтягиваются из него).  
3) Установите зависимости конкретного агента и запустите его.

### Запуск простого агента (`sgr-vanilla-agent-store`)
```
cd sgr-vanilla-agent-store
pip install -r requirements.txt
python main.py
```
Настройте модель и метаданные сессии в `sgr-vanilla-agent-store/main.py` (поля `MODEL_ID`, `workspace`, `name`).

### Запуск агента с памятью и смолкодом (`sgr-knowledge-agent-store`)
```
cd sgr-knowledge-agent-store
pip install -r requirements.txt
python main.py
```
По умолчанию `main.py` использует OpenAI. Поменяйте параметр `provider` в вызове `run_agent` на `"nebius"`, если хотите работать через Nebius (требуется `NEBIUS_API_KEY`).

## Полезно знать
- Агент сам стартует/завершает задачи через `ERC3` SDK; после выполнения сессия отправляется `submit_session`.
- Снимки консоли лежат в `res/console.png`; UI сессии — на https://erc.timetoact-group.at/.

## Поддержка
Вопросы по соревнованию — в Discord, ссылка приходит в письме после регистрации на ERC3.

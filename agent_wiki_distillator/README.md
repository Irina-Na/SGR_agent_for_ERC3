# Agent Wiki Distillator

Небольшой набор утилит для разметки и извлечения данных из вики:
- `wiki_annotator.py` — находит полезные файлы и формирует отчет `found.json`.
- `policy_extractor.py` — извлекает сущности по категориям из уже найденных файлов.

## Требования
- `OPENAI_API_KEY` в окружении.
- (опционально) `OPENAI_MODEL`, по умолчанию `gpt-4.1`.
- Дефолтный отчет: `agent_wiki_distillator/found_data/found-20251215T153124Z/found.json`.
- Дефолтный корень документов: `sgr-knowledge-agent-erc3_test/docs`.

## Извлечение одной категории
CLI минималистичный: первый аргумент — категория, второй (опционально) — путь к файлу для `apis`.

Доступные категории: `security_and_rules`, `locations`, `people_and_roles`, `systems_and_data`, `apis`, `all` (по умолчанию).

Примеры:
- Все категории (выгрузит отдельные JSON в `agent_wiki_extraction/found_data`):  
  ```bash
  python agent_wiki_distillator/policy_extractor.py
  ```
- Только люди и роли:  
  ```bash
  python agent_wiki_distillator/policy_extractor.py people_and_roles
  ```
- Только apis с дефолтным файлом (`sgr-knowledge-agent-erc3_test/docs/prep_desc.md`):  
  ```bash
  python agent_wiki_distillator/policy_extractor.py apis
  ```
- Только apis c явным файлом:  
  ```bash
  python agent_wiki_distillator/policy_extractor.py apis path/to/api_doc.md
  ```

Каждый запуск печатает путь(и) сохраненных JSON. Файлы лежат в `agent_wiki_extraction/found_data/{category}-<timestamp>.json`.

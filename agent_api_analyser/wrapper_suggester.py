from agent_api_analyser.executor import ExecutionResult


class WrapperSuggester:
    def suggest(self, results: list[ExecutionResult]) -> list[str]:
        ideas: list[str] = []
        for r in results:
            if r.ok:
                continue
            ideas.append(
                f"class Fix{r.scenario.request}: # handle {r.error or 'edge cases'}"
            )
        if not ideas:
            ideas.append("class PaginatedSearch: # normalize list/search paging")
        return ideas


from __future__ import annotations

import base64

from src.tools.web_search import WebSearchTool


def test_parse_bing_html_extracts_organic_results() -> None:
    html = """
    <ol id="b_results">
      <li class="b_algo">
        <h2><a href="https://example.org/paper">Evidence Paper</a></h2>
        <div class="b_caption"><p>A primary-source abstract.</p></div>
      </li>
    </ol>
    """

    assert WebSearchTool._parse_bing_html(html, 5) == [
        {
            "title": "Evidence Paper",
            "url": "https://example.org/paper",
            "snippet": "A primary-source abstract.",
        }
    ]


def test_unwrap_bing_redirect_url() -> None:
    target = "https://arxiv.org/abs/2509.13312"
    encoded = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    redirect = f"https://www.bing.com/ck/a?u=a1{encoded}&ntb=1"

    assert WebSearchTool._unwrap_bing_url(redirect) == target


def test_parse_brave_html_extracts_results() -> None:
    html = """
    <div class="snippet">
      <a href="https://arxiv.org/abs/2412.15115">
        <div class="title">Qwen2.5 Technical Report</div>
      </a>
      <div class="generic-snippet">Introduces Qwen2.5 and its 18T-token corpus.</div>
    </div>
    """

    assert WebSearchTool._parse_brave_html(html, 5) == [
        {
            "title": "Qwen2.5 Technical Report",
            "url": "https://arxiv.org/abs/2412.15115",
            "snippet": "Introduces Qwen2.5 and its 18T-token corpus.",
        }
    ]


def test_result_ranking_removes_irrelevant_bing_noise() -> None:
    results = [
        {
            "title": "Qwen2.5 Technical Report",
            "url": "https://arxiv.org/abs/2412.15115",
            "snippet": "Qwen2.5 large language model technical report",
        },
        {
            "title": "How to type a star symbol",
            "url": "https://example.com/star",
            "snippet": "Word keyboard tutorial",
        },
    ]

    ranked = WebSearchTool._rank_results("Qwen2.5 Technical Report 2412.15115", results, 5)

    assert [item["url"] for item in ranked] == ["https://arxiv.org/abs/2412.15115"]


def test_parse_yahoo_html_extracts_and_unwraps_results() -> None:
    html = """
    <div class="dd algo algo-sr">
      <div class="compTitle">
        <a href="https://r.search.yahoo.com/x/RU=https%3A%2F%2Farxiv.org%2Fabs%2F2412.15115/RK=2/RS=x">
          <h3 class="title">Qwen2.5 Technical Report</h3>
        </a>
      </div>
      <div class="compText"><p>Architecture and benchmark details for Qwen2.5.</p></div>
    </div>
    """

    assert WebSearchTool._parse_yahoo_html(html, 5) == [
        {
            "title": "Qwen2.5 Technical Report",
            "url": "https://arxiv.org/abs/2412.15115",
            "snippet": "Architecture and benchmark details for Qwen2.5.",
        }
    ]


def test_parse_wikipedia_response_marks_traceable_article_urls() -> None:
    data = {
        "query": {
            "search": [
                {
                    "title": "Qwen",
                    "snippet": "A family of <span class='searchmatch'>language models</span>.",
                }
            ]
        }
    }

    assert WebSearchTool._parse_wikipedia_response(data, "en", 5) == [
        {
            "title": "Qwen",
            "url": "https://en.wikipedia.org/wiki/Qwen",
            "snippet": "A family of language models .",
        }
    ]

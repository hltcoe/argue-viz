import argparse
import json
import os
import pandas as pd
import streamlit as st

from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nugget_data import AggregatorType, NuggetBank

ASSETS_DIR = Path(__file__).parent.parent / "assets"
EXAMPLE_TOPICS_JSONL = ASSETS_DIR / "neuclir24-test-request.jsonl"
EXAMPLE_NUGGETS_DIR = ASSETS_DIR / "example-nuggets"
EXAMPLE_JUDGMENTS_JSONL = (
    ASSETS_DIR
    / "example-report.judgments.jsonl"
)
EXAMPLE_SCORES_TSV = (
    ASSETS_DIR
    / "example-report.scores.tsv"
)
PER_TOPIC_DETAILED_METRICS = [
    "citation_relevance",
    "citation_support",
    "sentence_support",
    "nugget_coverage_weighted",
    "f1_weighted",
]
PER_TOPIC_SUMMARY_METRICS = [
    "sentence_support",
    "nugget_coverage_weighted",
    "f1_weighted",
]
PER_TOPIC_OTHER_STATS = [
    "sentences",
    "sentences_with_citations_or_requiring_citations",
    "correctly_cited_sentences",
    "character_count",
    "citations",
    "relevant_citations",
    "supporting_citations",
    "nuggets",
    "correct_nuggets",
    "nuggets_weighted",
    "correct_nuggets_weighted",
]

AGGREGATE_DETAILED_METRICS = [
    "citation_relevance_micro",
    "citation_relevance_macro",
    "citation_support_micro",
    "citation_support_macro",
    "sentence_support_micro",
    "sentence_support_macro",
    "nugget_coverage_weighted_micro",
    "nugget_coverage_weighted_macro",
    "f1_weighted_micro",
    "f1_weighted_macro",
]
AGGREGATE_SUMMARY_METRICS = [
    "sentence_support_micro",
    "sentence_support_macro",
    "nugget_coverage_weighted_micro",
    "nugget_coverage_weighted_macro",
    "f1_weighted_micro",
    "f1_weighted_macro",
]
ANNOTATIONS_HELP_TEXT = (
    "Each sentence of the report is shown on its own line below. "
    + "Sentences in :blue[blue] are judged to be supported by ***all*** their citations. "
    + "Sentences in :orange[orange] are unsupported by at least one citation. "
    + "The total citations and matched nuggets for each sentence are shown in brackets at the end of that sentence. "
    + "Click a sentence for more detailed information."
)
NUGGET_VIEW_HELP_TEXT = "Each tab below shows a matched nugget; only matched answers are shown within each tab. Citations (in gray) are citations for this sentence that attest the corresponding answer."
SUPPORT_VIEW_HELP_TEXT = "Each tab below shows a citation for this sentence. Citations shown in :blue[blue] are ones judged to support this sentence. Citations shown in :orange[orange] are ones judged *not* to support the sentence. **Bolded** citations are ones judged to be relevant."


def load_topics(topics_jsonl: Path) -> Dict[str, Any]:
    topics = {}
    with open(topics_jsonl, "r") as f:
        for line in f:
            topic_data = json.loads(line)
            topics[topic_data["request_id"]] = topic_data
    return topics


def load_nuggets(nuggets_dir: Path) -> Dict[str, NuggetBank]:
    all_nuggets = {}
    for nugget_file in glob(os.path.join(nuggets_dir, "*.json")):
        if not os.path.isfile(nugget_file):
            continue
        with open(nugget_file, "r") as f:
            nb = NuggetBank.model_validate(json.load(f))
            all_nuggets[nb.query_id] = nb
    return all_nuggets


def load_judgments(judgments_jsonl: Path) -> List[Dict[str, Any]]:
    judgments = []
    with open(judgments_jsonl, "r") as f:
        for line in f:
            judgments.append(json.loads(line))
    return judgments


def load_scores(scores_tsv: Path) -> pd.DataFrame:
    return pd.read_csv(scores_tsv, sep="\t")


def get_aggregate_results(scores: pd.DataFrame, detailed: bool = False) -> pd.DataFrame:

    def rename_metric(metric: str) -> str:
        if metric == "citation_support_micro":
            return "Citation Support"
        elif metric == "citation_support_macro":
            return "Citation Support"
        elif metric == "citation_relevance_micro":
            return "Citation Relevance"
        elif metric == "citation_relevance_macro":
            return "Citation Relevance"
        elif metric == "sentence_support_micro":
            return "Sentence Support"
        elif metric == "sentence_support_macro":
            return "Sentence Support"
        elif metric == "nugget_coverage_weighted_micro":
            return "Nugget Coverage"
        elif metric == "nugget_coverage_weighted_macro":
            return "Nugget Coverage"
        elif metric == "f1_weighted_micro":
            return "F1"
        elif metric == "f1_weighted_macro":
            return "F1"
        else:
            return metric

    columns = AGGREGATE_DETAILED_METRICS if detailed else AGGREGATE_SUMMARY_METRICS
    agg_results = scores[scores.metric.isin(columns) & (scores.request_id == "all")]
    agg_results["Metric Type"] = agg_results["metric"].str.split("_").str[-1]
    agg_results["metric"] = agg_results["metric"].apply(rename_metric)

    return agg_results


def get_per_topic_results(
    scores: pd.DataFrame, request_id: str, detailed: bool = False
) -> pd.DataFrame:

    def rename_metric(metric: str) -> str:
        if metric == "sentence_support":
            return "Sentence Support"
        elif metric == "citation_support":
            return "Citation Support"
        elif metric == "citation_relevance":
            return "Citation Relevance"
        elif metric == "nugget_coverage_weighted":
            return "Nugget Coverage"
        elif metric == "f1_weighted":
            return "F1"
        else:
            return metric

    columns = PER_TOPIC_DETAILED_METRICS if detailed else PER_TOPIC_SUMMARY_METRICS
    per_topic_results = scores[
        scores.metric.isin(columns) & (scores.request_id == request_id)
    ]
    per_topic_results["metric"] = per_topic_results["metric"].apply(rename_metric)

    return per_topic_results


def get_per_topic_other_stats(scores: pd.DataFrame, request_id: str) -> pd.DataFrame:
    stats_for_topic = scores[
        (scores.request_id == request_id) & (scores.metric.isin(PER_TOPIC_OTHER_STATS))
    ]
    stats_for_topic["metric"] = stats_for_topic.metric.str.replace("_", " ")
    stats_for_topic = stats_for_topic.rename(
        columns={"metric": "Statistic", "value": "Value"}
    )
    return stats_for_topic[["Statistic", "Value"]]


def get_run_id_from_scores(scores: pd.DataFrame) -> str:
    if "run_id" in scores.columns:
        return scores["run_id"].iloc[0]
    else:
        raise ValueError("The scores DataFrame does not contain a 'run_id' column.")


def get_citation_colored(
    citation: str,
    citation_id: str,
    citation_support: Dict[str, bool],
    citation_relevance: Dict[str, bool],
) -> str:
    if citation_support.get(citation, False):
        if citation_relevance.get(citation, False):
            return f"**:blue[c{citation_id}]**"
        else:
            return f":blue[c{citation_id}]"
    else:
        if citation_relevance.get(citation, False):
            return f"**:orange[c{citation_id}]**"
        else:
            return f":orange[c{citation_id}]"


def get_doc_title_and_text(text: str) -> Tuple[Optional[str], str]:
    if text.startswith("Title:"):
        text_pieces = text.split("\n\n")
        title = text_pieces[0][7:]
        if text_pieces[1].startswith("Content:"):
            text_pieces[1] = text_pieces[1][9:]
        content = "\n\n".join(text_pieces[1:])
    else:
        title = None
        content = text
    return title, content


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Streamlit app.")
    parser.add_argument(
        "--topic-data",
        default=EXAMPLE_TOPICS_JSONL,
        type=str,
    )
    parser.add_argument(
        "--nuggets-dir",
        default=EXAMPLE_NUGGETS_DIR,
        type=str,
    )
    parser.add_argument(
        "--judgments-data",
        default=EXAMPLE_JUDGMENTS_JSONL,
        type=str,
    )
    parser.add_argument(
        "--scores-data",
        default=EXAMPLE_SCORES_TSV,
        type=str,
    )
    args = parser.parse_args()

    assert os.path.isfile(
        args.topic_data
    ), f"Topic data file {args.topic_data} does not exist!"
    topic_data = load_topics(Path(args.topic_data))

    assert os.path.isdir(
        args.nuggets_dir
    ), f"Nuggets directory {args.nuggets_dir} does not exist!"
    nugget_data = load_nuggets(Path(args.nuggets_dir))

    assert os.path.isfile(
        args.judgments_data
    ), f"Judgments file {args.judgments_data} does not exist!"
    judgment_data = load_judgments(Path(args.judgments_data))

    assert os.path.isfile(
        args.scores_data
    ), f"Scores file {args.scores_data} does not exist!"
    scores = load_scores(scores_tsv=Path(args.scores_data))

    topic_to_judgment_data = {
        jd["report"]["metadata"]["topic_id"]: jd for jd in judgment_data
    }
    team_id = {jd["report"]["metadata"]["team_id"] for jd in judgment_data}
    assert len(team_id) == 1, f"Multiple team IDs found!"
    team_id = list(team_id)[0]
    agg_results = get_aggregate_results(scores)
    run_id = get_run_id_from_scores(scores)

    with st.sidebar:
        topic_ids = sorted(
            [jd["report"]["metadata"]["topic_id"] for jd in judgment_data]
        )
        topic_titles = [topic_data[t]["title"] for t in topic_ids if t != "all"]
        chosen_topic = st.radio(
            "**Choose a topic:**", ["all"] + topic_ids, captions=[""] + topic_titles
        )

    ## Aggregate statistics across all topics
    if chosen_topic == "all":
        st.divider()
        st.header(":primary[All Topics]")
        st.divider()
        st.header("Scores")
        st.caption(
            f"ARGUE scores aggregated across all topics for team **:primary[{team_id}]** and run **:primary[{run_id}]**."
        )
        core, detailed = st.tabs(["Core Metrics", "Detailed Metrics"])

        # "Core" summary metrics
        with core:
            agg_results_core = get_aggregate_results(scores)
            st.bar_chart(
                agg_results_core,
                x="metric",
                y="value",
                x_label="Score",
                y_label="Metric",
                color="Metric Type",
                horizontal=True,
                stack=False,
            )

        # More detailed metrics (adds citation support + citation relevance)
        with detailed:
            agg_results_detailed = get_aggregate_results(scores, detailed=True)
            st.bar_chart(
                agg_results_detailed,
                x="metric",
                y="value",
                x_label="Score",
                y_label="Metric",
                color="Metric Type",
                horizontal=True,
                stack=False,
            )

    ## Per-topic metrics and annotations
    else:
        # Topic Info (problem statement and background)
        st.divider()
        st.header(
            f":primary[Topic {chosen_topic}:] {topic_data[chosen_topic]["title"]}"
        )
        st.divider()
        st.subheader("Problem Statement")
        st.write(topic_data[chosen_topic]["problem_statement"])
        st.subheader("Background")
        st.write(topic_data[chosen_topic]["background"])
        st.divider()

        # Metrics
        st.header("Scores")
        st.caption(
            f"ARGUE scores on topic **:primary[{chosen_topic}]** for team **:primary[{team_id}]** and run **:primary[{run_id}]**."
        )
        core, detailed, other = st.tabs(
            ["Core Metrics", "Detailed Metrics", "Other Statistics"]
        )

        # "Core" summary metrics
        with core:
            topic_results_core = get_per_topic_results(scores, chosen_topic)
            st.bar_chart(
                topic_results_core,
                x="metric",
                y="value",
                x_label="Score",
                y_label="Metric",
                horizontal=True,
                stack=False,
            )

        # More detailed metrics (adds citation support + citation relevance)
        with detailed:
            topic_results_detailed = get_per_topic_results(
                scores, chosen_topic, detailed=True
            )
            st.bar_chart(
                topic_results_detailed,
                x="metric",
                y="value",
                x_label="Score",
                y_label="Metric",
                horizontal=True,
                stack=False,
            )

        # Other statistics (e.g., number of sentences, citations, nuggets)
        with other:
            topic_other_stats = get_per_topic_other_stats(
                scores, request_id=chosen_topic
            )
            st.dataframe(topic_other_stats, hide_index=True)

        ## Annotations Info
        st.divider()
        st.header("Annotations")
        st.caption(ANNOTATIONS_HELP_TEXT)

        judgment_data_for_report = topic_to_judgment_data[chosen_topic]
        sent_citations_summary = []
        sent_texts_summary = []
        sent_texts_detailed = []
        explanations = []
        all_matched_nuggets = []
        all_citation_tab_names = []
        all_citation_titles = []
        all_citation_texts = []
        all_citation_doc_ids = []
        nugget_q_to_id = {
            q: i for i, q in enumerate(sorted(nugget_data[chosen_topic].nugget_bank))
        }
        nugget_id_to_q = {i: q for q, i in nugget_q_to_id.items()}
        nugget_count = 0
        matched_nuggets_for_report = defaultdict(lambda: defaultdict(set))
        citation_to_info = {}
        citation_to_id = {}
        id_to_citation = {}
        citation_count = 0
        unique_citation_count = 0

        # Extract info from judgment data. Note that we
        # do not explicitly represent judgment objects in the interface,
        # but rather try to represent the information they contain visually
        # (e.g. via color coding)
        for i, judgment_data_for_sent in enumerate(
            judgment_data_for_report["judgments"]
        ):
            citation_ids = []
            citation_titles = []
            citation_texts = []
            for cit in judgment_data_for_sent["citations"]:
                citation_ids.append(cit["doc_id"])
                title, content = get_doc_title_and_text(cit["text"])
                citation_titles.append(title)
                citation_texts.append(content)
            citation_support = {}
            citation_relevance = {}
            matched_nuggets_for_sent = defaultdict(lambda: defaultdict(set))
            sent_attested_by_all_citations = True
            for judgment in judgment_data_for_sent["judgments"]:
                if judgment["judgment_type_id"] == "SENTENCE_ATTESTED":
                    if judgment["response"]["is_attested"]:
                        citation_support[judgment["provenance"]["doc_id"]] = True
                    else:
                        citation_support[judgment["provenance"]["doc_id"]] = False
                        sent_attested_by_all_citations = False
                elif judgment["judgment_type_id"] == "CITED_DOCUMENT_RELEVANCE":
                    if judgment["response"]["is_relevant"]:
                        citation_relevance[judgment["provenance"]["doc_id"]] = True
                    else:
                        citation_relevance[judgment["provenance"]["doc_id"]] = False
                elif judgment["judgment_type_id"] == "SENTENCE_ANSWERS_QUESTION":
                    for nug in judgment["response"]["matched_nuggets"]:
                        if nug["question_text"] not in nugget_q_to_id:
                            nugget_q_to_id[nug["question_text"]] = nugget_count
                            nugget_id_to_q[nugget_count] = nug["question_text"]
                            nugget_count += 1
                        for ans in nug["matched_answer"]:
                            matched_nuggets_for_sent[nug["question_text"]][ans].add(
                                judgment["provenance"]["doc_id"]
                            )
                            matched_nuggets_for_report[nug["question_text"]][ans].add(
                                judgment["provenance"]["doc_id"]
                            )
            for cit in judgment_data_for_sent["citations"]:
                citation_count += 1
                if cit["doc_id"] not in citation_to_id:
                    title, content = get_doc_title_and_text(cit["text"])
                    citation_to_info[cit["doc_id"]] = (title, content)
                    citation_to_id[cit["doc_id"]] = unique_citation_count
                    id_to_citation[unique_citation_count] = cit["doc_id"]
                    unique_citation_count += 1

            all_matched_nuggets.append(
                sorted(
                    [
                        (nugget_q_to_id[nq], nq, nas)
                        for (nq, nas) in matched_nuggets_for_sent.items()
                    ]
                )
            )
            all_citation_tab_names.append(
                [
                    get_citation_colored(
                        cit, citation_to_id[cit], citation_support, citation_relevance
                    )
                    for cit in citation_ids
                ]
            )
            all_citation_titles.append(citation_titles)
            all_citation_texts.append(citation_texts)
            all_citation_doc_ids.append(citation_ids)
            if sent_attested_by_all_citations:
                sent_text = f":blue[{judgment_data_for_sent['text']}]"
            else:
                sent_text = f":orange[{judgment_data_for_sent['text']}]"
            if len(citation_texts) == 1:
                citation_str = "citation"
            else:
                citation_str = "citations"
            if len(matched_nuggets_for_sent) == 1:
                nuggets_str = "nugget"

            else:
                nuggets_str = "nuggets"
            sent_citations_summary.append(
                "**["
                + ", ".join([f"c{citation_to_id[cid]}" for cid in citation_ids])
                + "]**"
            )
            sent_texts_summary.append(sent_text)
            sent_texts_detailed.append(
                f"**[S{i+1}]:** {sent_text}"
                + f" **[{len(citation_texts)} {citation_str}; {len(matched_nuggets_for_sent)} {nuggets_str}]**"
            )

        (
            report_view_tab,
            sentence_view_tab,
        ) = st.tabs(["Report View", "Sentence View"])

        ## Report-level view
        with report_view_tab:
            st.write(
                f"**Report**: {' '.join([sts + ' ' + scs for (sts, scs) in zip(sent_texts_summary, sent_citations_summary)])}"
            )
            citations_tab, nuggets_tab = st.tabs(["Citations", "Nuggets"])

            # Information about citation support
            with citations_tab:
                unique_citations = sorted(
                    citation_to_id[cid] for cid in citation_to_info.keys()
                )
                st.caption(
                    f"Total citations: **:primary[{citation_count}]**. Unique citations: **:primary[{unique_citation_count}]**."
                )
                doc_tabs = st.tabs([f"**c{cid}**" for cid in unique_citations])
                for cid, tab in zip(unique_citations, doc_tabs):
                    doc_id = id_to_citation[cid]
                    title, text = citation_to_info[doc_id]
                    with tab:
                        st.write(f"**Doc ID**: {doc_id}")
                        if title:
                            st.write(f"**Title**: {title}")
                        st.write(f"**Text**: {text}")

            # Information about matched nuggets
            with nuggets_tab:
                # nugget_ids = sorted([nugget_q_to_id[q] for q in matched_nuggets_for_report])
                nugget_ids = sorted(nugget_id_to_q.keys())
                nugget_tab_names = []
                correct_nuggets = 0
                for nid in nugget_ids:
                    nq = nugget_id_to_q[nid]
                    if nq in matched_nuggets_for_report:
                        aggregator_type = (
                            nugget_data[chosen_topic].nugget_bank[nq].aggregator_type
                        )
                        # We matched at least one answer, so we're good
                        if aggregator_type == AggregatorType.OR:
                            correct_nuggets += 1
                            nugget_tab_names.append(f"**:blue[n{nid}]**")
                        elif aggregator_type == AggregatorType.AND:
                            matched_answers = set(matched_nuggets_for_report[nq])
                            gold_answers = set(
                                nugget_data[chosen_topic].nugget_bank[nq].answers
                            )
                            if matched_answers == gold_answers:
                                correct_nuggets += 1
                                nugget_tab_names.append(f"**:blue[n{nid}]**")
                            else:
                                nugget_tab_names.append(f"**:orange[n{nid}]**")
                    else:
                        nugget_tab_names.append(f"**:orange[n{nid}]**")
                st.caption(
                    f"Total nuggets: **:primary[{len(nugget_data[chosen_topic].nugget_bank)}]**. Total nuggets matched by report: **:primary[{len(matched_nuggets_for_report)}]**. Total correct nuggets: **:primary[{correct_nuggets}]**."
                )
                st.caption(
                    f"All correct nuggets are shown below in **:blue[blue]** and all incorrect nuggets are shown in **:orange[orange]**."
                )
                nugget_tabs = st.tabs(nugget_tab_names)
                for nid, tab in zip(nugget_ids, nugget_tabs):
                    with tab:
                        nq = nugget_id_to_q[nid]
                        st.write(f"**Question:** {nq}")
                        st.write(
                            f"**Type**: {nugget_data[chosen_topic].nugget_bank[nq].aggregator_type.value}"
                        )
                        if nq in matched_nuggets_for_report:
                            st.write(f"**Matched Answers**:")
                            for ans, citations in matched_nuggets_for_report[
                                nq
                            ].items():
                                st.badge(ans)
                                citation_ids = sorted(
                                    [citation_to_id[c] for c in citations]
                                )
                                st.caption(
                                    "["
                                    + ", ".join([f"**c{cid}**" for cid in citation_ids])
                                    + "]"
                                )
                        else:
                            st.write(f"**Matched Answers:** None")
                        unmatched_answers = []
                        for ans in nugget_data[chosen_topic].nugget_bank[nq].answers:
                            if ans not in matched_nuggets_for_report[nq]:
                                unmatched_answers.append(ans)
                        if unmatched_answers:
                            st.write(f"**Unmatched Answers**:")
                            for ans in unmatched_answers:
                                st.badge(ans, color="orange")
                        else:
                            st.write(f"**Unmatched Answers:** None")

        ## Sentence-level view
        with sentence_view_tab:
            sent_text_expanders = [
                st.expander(sent_text) for sent_text in sent_texts_detailed
            ]
            for (
                expander,
                matched_nuggets,
                citation_tab_names,
                citation_titles,
                citation_texts,
                citation_doc_ids,
            ) in zip(
                sent_text_expanders,
                all_matched_nuggets,
                all_citation_tab_names,
                all_citation_titles,
                all_citation_texts,
                all_citation_doc_ids,
            ):
                with expander:
                    support_view, nugget_view = st.tabs(["Support", "Nuggets"])

                    # Sentence-level support information
                    with support_view:
                        st.caption(SUPPORT_VIEW_HELP_TEXT)
                        citation_tabs = st.tabs(citation_tab_names)
                        for tab, title, text, doc_id in zip(
                            citation_tabs,
                            citation_titles,
                            citation_texts,
                            citation_doc_ids,
                        ):
                            with tab:
                                st.write(f"**Doc ID**: {doc_id}")
                                if title is not None:
                                    st.write(f"**Title**: {title}")
                                st.write(f"**Text**: {text}")

                    # Sentence-level nugget information
                    with nugget_view:
                        st.caption(NUGGET_VIEW_HELP_TEXT)
                        if matched_nuggets:
                            nugget_tab_names = []
                            for n in matched_nuggets:
                                nq = n[1]
                                if nq in nugget_q_to_id:
                                    nid = nugget_q_to_id[nq]
                                    aggregator_type = (
                                        nugget_data[chosen_topic]
                                        .nugget_bank[nq]
                                        .aggregator_type
                                    )
                                    if aggregator_type == AggregatorType.OR:
                                        nugget_tab_names.append(f"**:blue[n{nid}]**")
                                    elif aggregator_type == AggregatorType.AND:
                                        matched_answers = set(n[2].keys())
                                        gold_answers = set(
                                            nugget_data[chosen_topic]
                                            .nugget_bank[nq]
                                            .answers
                                        )
                                        if matched_answers == gold_answers:
                                            nugget_tab_names.append(
                                                f"**:blue[n{nid}]**"
                                            )
                                        else:
                                            nugget_tab_names.append(
                                                f"**:orange[n{nid}]**"
                                            )
                                else:
                                    nugget_tab_names.append(f"**:orange[n{n[0]}]**")
                            nugget_tabs = st.tabs(nugget_tab_names)
                            for tab, nug in zip(nugget_tabs, matched_nuggets):
                                with tab:
                                    st.write(f"**Question:** {nug[1]}")
                                    st.write(
                                        f"**Type**: {nugget_data[chosen_topic].nugget_bank[nug[1]].aggregator_type.value}"
                                    )
                                    st.write(f"**Matched Answers**:")
                                    for ans, citations in nug[2].items():
                                        st.badge(ans)
                                        st.caption(
                                            "["
                                            + ", ".join(
                                                [
                                                    f"**c{citation_to_id[c]}**"
                                                    for c in citations
                                                ]
                                            )
                                            + "]"
                                        )
                                    gold_answers = (
                                        nugget_data[chosen_topic]
                                        .nugget_bank[nug[1]]
                                        .answers
                                    )
                                    unmatched_answers = set(gold_answers) - set(
                                        nug[2].keys()
                                    )
                                    if unmatched_answers:
                                        st.write(f"**Unmatched Answers**:")
                                        for ans in unmatched_answers:
                                            st.badge(ans, color="orange")
                                    else:
                                        st.write(f"**Unmatched Answers:** None")

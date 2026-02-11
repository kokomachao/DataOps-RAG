from __future__ import annotations

import typer
import uvicorn

from rag.data.build_dataset import build_dataset_from_posts_xml
from rag.index import build_all

cli = typer.Typer(help="Data Platform RAG Assistant CLI")

@cli.command("build-dataset")
def build_dataset(
    posts: str = typer.Option(..., help="Path to Posts.xml"),
    out: str = typer.Option(..., help="Output JSONL path"),
    components: list[str] = typer.Option(["spark","flink","kafka","hadoop","hive"], help="Target components"),
    max_questions: int = typer.Option(200000, help="Max questions to scan"),
    min_score: int = typer.Option(-5, help="Min score threshold"),
):
    n = build_dataset_from_posts_xml(posts, out, components=components, max_questions=max_questions, min_score=min_score)
    typer.echo(f"Wrote {n} QA records to {out}")


@cli.command("build-dataset-csv")
def build_dataset_csv(
    questions: str = typer.Option(..., help="Path to Questions.csv"),
    answers: str = typer.Option(..., help="Path to Answers.csv"),
    tags: str = typer.Option(..., help="Path to Tags.csv"),
    out: str = typer.Option(..., help="Output JSONL path"),
    components: list[str] = typer.Option(["spark","flink","kafka","hadoop","hive"], help="Target components"),
    max_questions: int = typer.Option(None, help="Optional cap of candidate questions for quick tests"),
    min_q_score: int = typer.Option(-5, help="Min question score"),
    min_a_score: int = typer.Option(-5, help="Min answer score"),
):
    from rag.data.build_dataset_csv import build_dataset_from_csv_tables
    n = build_dataset_from_csv_tables(
        questions_csv=questions,
        answers_csv=answers,
        tags_csv=tags,
        out_jsonl=out,
        components=components,
        max_questions=max_questions,
        min_q_score=min_q_score,
        min_a_score=min_a_score,
    )
    typer.echo(f"Wrote {n} QA records to {out}")

@cli.command("build-index")
def build_index(
    data: str = typer.Option(..., help="Processed JSONL"),
    backend: str = typer.Option("faiss", help="faiss|milvus"),
    storage: str = typer.Option("storage", help="Storage directory"),
    chunk_size: int = typer.Option(900, help="Chunk size"),
    chunk_overlap: int = typer.Option(150, help="Chunk overlap"),
):
    meta = build_all(data, backend=backend.lower(), storage_dir=storage, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    typer.echo("Index build done.")
    typer.echo(meta)

@cli.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", help="Host"),
    port: int = typer.Option(8000, help="Port"),
):
    uvicorn.run("app.main:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    cli()


import polars as pl
import networkx as nx

def create_nx_graph(df: pl.DataFrame) -> nx.DiGraph:
    """Build a directed comment thread graph with edges pointing parent to child.

    Each node is a comment ID or a submission ID (the Reddit type prefix such
    as 't1_' or 't3_' is stripped). An edge (parent -> comment) means the
    comment is a direct reply to the parent. This produces a forest of trees,
    one tree per submission thread.

    Args:
        df: DataFrame with 'id', 'parent_id', and 'link_id' columns.
            All three columns must be in Reddit's 'tX_<id>' format.

    Returns:
        Directed acyclic graph suitable for top-down thread traversal.
        Submission root nodes (derived from 'link_id') are included as
        nodes with no incoming edges.

    Note:
        The edge direction here is parent -> child (oldest ancestor to newest
        reply), which is the natural reading order of a conversation. This
        allows straightforward DFS/BFS traversal in thread_export functions.
    """
    G = nx.DiGraph()
    G.add_nodes_from(df["id"])
    
    # Add submission root nodes (strip the 't3_' type prefix)
    link_roots = (
        df["link_id"]
        .str.split("_")
        .list.get(1)
        .fill_null("unknown")
    )
    G.add_nodes_from(link_roots.unique())
    
    # Extract and add edges: parent_id (stripped) -> comment id
    df = df.with_columns(
        pl.col("parent_id")
        .str.split("_")
        .list.get(1, null_on_oob=True)
        .fill_null("unknown")
        .alias("parent_id_stripped")
    )
    
    G.add_edges_from(
        zip(
            df.filter(pl.col('parent_id_stripped') != "unknown")["parent_id_stripped"],
            df.filter(pl.col('parent_id_stripped') != "unknown")["id"],
        )
    )
    
    return G


def get_parent_author(df: pl.DataFrame) -> pl.DataFrame:
    """Add a 'parent_author' column mapping each comment to its parent's author.

    Looks up each comment's parent_id in the same DataFrame. Comments whose
    parent is the submission root (parent_id starts with 't3_') will have
    null in 'parent_author' since submissions are not rows in the DataFrame.

    Args:
        df: DataFrame with at least 'id', 'author', and 'parent_id' columns.
            The 'parent_id' column must be in Reddit's 'tX_<id>' format.

    Returns:
        A copy of the input DataFrame with a new 'parent_author' column.
        The original DataFrame is not modified.
    """
    df = df.with_columns(
        pl.col("parent_id")
        .str.split("_")
        .list.get(1, null_on_oob=True)
        .alias("parent_id_stripped")
    )
    df = df.join(
        df.select(pl.col("id"), pl.col("author").alias("parent_author")),
        left_on="parent_id_stripped",
        right_on="id",
        how="left"
    )
    df = df.drop(["parent_id_stripped"])
    return df


def extract_interaction_graph(comment_df: pl.DataFrame) -> nx.DiGraph:
    """Build a weighted directed user interaction graph from a comments DataFrame.

    Each node is a Reddit username. A directed edge (u -> v) with weight w
    means user u replied to user v's comments w times. Self-loops are
    included when a user replies to themselves.

    Args:
        comment_df: DataFrame with at least 'author', 'parent_author', and
            'id' columns. The 'parent_author' column should be pre-computed
            via get_parent_author(). Rows where 'parent_author' is null
            (top-level comments replying to a submission) are dropped.

    Returns:
        Directed graph with 'weight' edge attributes representing reply counts.
    """
    user_interactions = (
        comment_df.filter(pl.col("parent_author").is_not_null())
        .group_by(["author", "parent_author"])
        .agg(pl.col("id").count().alias("count"))
    )
    G = nx.DiGraph()
    G.add_edges_from(
        zip(
            user_interactions["author"],
            user_interactions["parent_author"],
            [{"weight": c} for c in user_interactions["count"]],
        )
    )
    return G
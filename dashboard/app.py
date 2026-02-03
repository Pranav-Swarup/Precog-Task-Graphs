"""
metafam dashboard
run with streamlit run app.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import os
import sys

# path setup maybe fragile
# print("booting app path config")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_backend import DashboardData
from visualization import GraphViz

st.set_page_config(
    page_title="MetaFAM Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.caption("Made by Pranav Swarup Kumar")

# session state init
if 'data' not in st.session_state:
    st.session_state.data = None
if 'viz' not in st.session_state:
    st.session_state.viz = None

st.sidebar.title("MetaFAM Explorer")
st.sidebar.markdown("---")

data_path = st.sidebar.text_input(
    "Data file path",
    value="./data/train.txt",
    help="path to triplets file"
)

col1, col2 = st.sidebar.columns(2)

if col1.button("Load Data"):
    # print("load data clicked")
    if os.path.exists(data_path):
        with st.spinner("loading..."):
            data = DashboardData()
            data.load(data_path)
            st.session_state.data = data
            st.session_state.viz = GraphViz(data)
        st.sidebar.success(f"loaded {len(data.people)} people")
    else:
        st.sidebar.error(f"file not found: {data_path}")

if col2.button("Load Cache"):
    # print("attempting cache load")
    data = DashboardData()
    if data.load_cache():
        st.session_state.data = data
        st.session_state.viz = GraphViz(data)
        st.sidebar.success("loaded from cache")
    else:
        st.sidebar.warning("no cache found")

if st.session_state.data is None:
    st.title("MetaFAM Knowledge Graph Explorer")
    st.info("load data from sidebar to begin")
    # print("no data stopping early")
    st.stop()

data = st.session_state.data
viz = st.session_state.viz

tab_overview, tab_explore, tab_pathfinder, tab_person, tab_stats = st.tabs([
    "Overview",
    "Graph Explorer",
    "Path Finder",
    "Person Lookup",
    "Statistics"
])

# ---------------- overview ----------------

with tab_overview:
    st.header("Dataset Overview")

    full_stats = data.get_full_graph_stats()
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    c1.metric("People", len(data.people))
    c2.metric("Relations", len(data.triplets))
    c3.metric("Relation Types", len(data.relation_types))
    c4.metric("Families", full_stats['num_families'])
    c5.metric("Avg Degree", f"{full_stats['avg_degree']:.1f}")
    c6.metric("Max Degree", full_stats['max_degree'])

    gen_stats = data.get_generation_stats()
    max_gen = max(k for k in gen_stats.keys() if k >= 0) if any(k >= 0 for k in gen_stats.keys()) else 0

    c1_2, c2_2, c3_2 = st.columns(3)
    c1_2.metric("Generations", f"0-{max_gen}")
    founders = data.get_founders()
    c2_2.metric("Founders", len(founders))
    anomalous = data.get_anomalous_nodes()
    c3_2.metric("Anomalies", len(anomalous))

    st.markdown("---")

    st.subheader("Find Interesting People")

    interest_cols = st.columns(5)

    with interest_cols[0]:
        if st.button("Highest Degree", use_container_width=True):
            st.session_state.interesting_metric = 'degree'
    with interest_cols[1]:
        if st.button("Highest Betweenness", use_container_width=True):
            st.session_state.interesting_metric = 'betweenness'
    with interest_cols[2]:
        if st.button("Top Founders", use_container_width=True):
            st.session_state.interesting_metric = 'founders'
    with interest_cols[3]:
        if st.button("Gen Z-Score", use_container_width=True):
            st.session_state.interesting_metric = 'gen_zscore'
    with interest_cols[4]:
        if st.button("Bridge Nodes", use_container_width=True):
            st.session_state.interesting_metric = 'bridges'

    if 'interesting_metric' in st.session_state:
        metric = st.session_state.interesting_metric
        st.write(f"Top 15 by {metric}")

        with st.spinner("Computing..."):
            interesting = data.get_interesting_people(metric, n=15)

        if interesting:
            df = pd.DataFrame(interesting)

            if metric == 'degree':
                cols = ['person_id', 'degree', 'generation', 'gender', 'num_children']
            elif metric == 'betweenness':
                cols = ['person_id', 'betweenness', 'degree', 'generation', 'gender']
            elif metric == 'founders':
                cols = ['person_id', 'descendants', 'num_children', 'generation', 'gender']
            elif metric == 'gen_zscore':
                cols = ['person_id', 'zscore', 'degree', 'gen_avg_degree', 'generation']
            elif metric == 'bridges':
                cols = ['person_id', 'betweenness', 'degree', 'generation', 'gender']
            else:
                cols = list(df.columns)[:6]

            cols = [c for c in cols if c in df.columns]
            st.dataframe(df[cols], use_container_width=True)
        else:
            st.info("no results found")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        gen_df = pd.DataFrame(
            [{'Generation': k, 'Count': v} for k, v in gen_stats.items() if k >= 0]
        )
        fig = px.bar(
            gen_df,
            x='Generation',
            y='Count',
            title='Generation Distribution',
            color='Generation',
            color_continuous_scale='viridis'
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_stats = data.get_gender_stats()
        gender_df = pd.DataFrame(
            [{'Gender': k, 'Count': v} for k, v in gender_stats.items()]
        )
        fig = px.pie(
            gender_df,
            values='Count',
            names='Gender',
            title='Gender Distribution',
            color='Gender',
            color_discrete_map={
                'F': '#e91e8c',
                'M': '#2196f3',
                'Unknown': '#9e9e9e'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Relation Types")
    rel_stats = data.get_relation_stats()

    if isinstance(rel_stats, dict):
        rel_stats = rel_stats.items()

    top_relations = sorted(rel_stats, key=lambda x: x[1], reverse=True)[:15]
    rel_df = pd.DataFrame(top_relations, columns=["Relation", "Count"])

    fig = px.bar(
        rel_df,
        x='Relation',
        y='Count',
        title='Top 15 Relation Types',
        color='Count',
        color_continuous_scale='blues'
    )
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# ---------------- graph explorer ----------------

with tab_explore:
    st.header("Interactive Graph")
    
    # controls in sidebar when this tab is active
    st.sidebar.markdown("---")
    st.sidebar.subheader("Graph Controls")
    
    view_mode = st.sidebar.radio(
        "View Mode",
        ["Family Tree","Ego Network", "Sample Nodes"],
        index=0,
    )
    
    color_by = st.sidebar.selectbox(
        "Color by",
        ["generation", "gender", "degree"],
    )
    
    size_by = st.sidebar.selectbox(
        "Size by",
        ["degree", "children", "fixed"],
    )
    
    if view_mode == "Ego Network":
        # person selector
        st.subheader("Ego Network")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            person_search = st.text_input("Search person", value="olivia0")
        with col2:
            hops = st.selectbox("Hops", [1, 2, 3], index=0)
        
        # quick picks
        st.write("Quick picks:")
        quick_cols = st.columns(5)
        quick_picks = ['olivia0', 'katharina1', 'fabian26', 'emma7', 'jonas23']
        for i, qp in enumerate(quick_picks):
            if quick_cols[i].button(qp, key=f"qp_{qp}"):
                person_search = qp
        
        if person_search and person_search in data.people:
            subgraph = data.get_ego_network(person_search, hops)
            
            # subgraph stats comparison
            sub_stats = data.get_subgraph_stats(subgraph['nodes'])
            full_stats = data.get_full_graph_stats()
            
            with st.expander("📊 Subgraph Stats (vs Full Graph)", expanded=False):
                stat_cols = st.columns(4)
                stat_cols[0].metric("Nodes", sub_stats['num_nodes'], 
                                   f"{sub_stats['num_nodes']/full_stats['num_nodes']*100:.2f}%")
                stat_cols[1].metric("Edges", sub_stats['num_edges'],
                                   f"{sub_stats['num_edges']/full_stats['num_edges']*100:.2f}%")
                stat_cols[2].metric("Avg Degree", f"{sub_stats['avg_degree']:.1f}",
                                   f"full: {full_stats['avg_degree']:.1f}")
                stat_cols[3].metric("Founders", sub_stats['num_founders'])
                
                st.write(f"**Gender:** M={sub_stats['gender_counts'].get('M', 0)}, "
                        f"F={sub_stats['gender_counts'].get('F', 0)}, "
                        f"Unknown={sub_stats['gender_counts'].get('Unknown', 0)}")
                st.write(f"**Generation Range:** {sub_stats['generation_range'][0]} - {sub_stats['generation_range'][1]}")
            
            st.info(f"Showing {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges")
            
            net = viz.create_graph(
                subgraph['nodes'],
                subgraph['edges'],
                color_by=color_by,
                size_by=size_by,
            )
            
            # save and display
            net.save_graph("dashboard/temp_graph.html")
            with open("dashboard/temp_graph.html", "r") as f:
                html = f.read()
            st.components.v1.html(html, height=700, scrolling=True)
        else:
            st.warning(f"Person '{person_search}' not found")
    
    elif view_mode == "Family Tree":
        st.subheader("Family Tree View")

        col1, col2 = st.columns([3, 1])
        with col1:
            person_search = st.text_input("Search person", value="olivia0", key="tree_search")
            show_all_edges = st.toggle(
                "Show all relations (importance view)",
                value=False,
                help="Include all incoming and outgoing relations for this person"
            )
        with col2:
            st.write("")

        st.write("Quick picks:")
        quick_cols = st.columns(5)
        quick_picks = ['olivia0', 'dominik1036', 'fabian26', 'emma7', 'jonas23']
        for i, qp in enumerate(quick_picks):
            if quick_cols[i].button(qp, key=f"tree_qp_{qp}"):
                person_search = qp

        if person_search and person_search in data.people:
            tree = data.get_family_tree(person_search)

            if show_all_edges:
                extra_nodes = set()
                extra_edges = []

                # outgoing edges
                for r, t in data.outgoing[person_search]:
                    extra_nodes.add(t)
                    extra_edges.append({
                        'from': person_search,
                        'relation': r,
                        'to': t
                    })

                # incoming edges
                for r, s in data.incoming[person_search]:
                    extra_nodes.add(s)
                    extra_edges.append({
                        'from': s,
                        'relation': r,
                        'to': person_search
                    })

                # merge
                tree['nodes'] = list(set(tree['nodes']) | extra_nodes)
                tree['edges'].extend(extra_edges)

            # subgraph stats comparison
            sub_stats = data.get_subgraph_stats(tree['nodes'])
            full_stats = data.get_full_graph_stats()
            
            with st.expander("📊 Subgraph Stats (vs Full Graph)", expanded=False):
                stat_cols = st.columns(4)
                stat_cols[0].metric("Nodes", sub_stats['num_nodes'], 
                                   f"{sub_stats['num_nodes']/full_stats['num_nodes']*100:.2f}%")
                stat_cols[1].metric("Edges", sub_stats['num_edges'],
                                   f"{sub_stats['num_edges']/full_stats['num_edges']*100:.2f}%")
                stat_cols[2].metric("Avg Degree", f"{sub_stats['avg_degree']:.1f}",
                                   f"full: {full_stats['avg_degree']:.1f}")
                stat_cols[3].metric("Founders", sub_stats['num_founders'])
                
                st.write(f"**Gender:** M={sub_stats['gender_counts'].get('M', 0)}, "
                        f"F={sub_stats['gender_counts'].get('F', 0)}, "
                        f"Unknown={sub_stats['gender_counts'].get('Unknown', 0)}")
                st.write(f"**Generation Range:** {sub_stats['generation_range'][0]} - {sub_stats['generation_range'][1]}")

            st.info(f"Family tree: {len(tree['nodes'])} members")

            net = viz.create_graph(
                tree['nodes'],
                tree['edges'],
                color_by=color_by,
                size_by='fixed',
                physics=False,
            )

            net.save_graph("dashboard/temp_graph.html")
            with open("dashboard/temp_graph.html", "r") as f:
                html = f.read()
            st.components.v1.html(html, height=700, scrolling=True)
        else:
            st.warning(f"Person '{person_search}' not found")

    
    else:  # sample nodes
        st.subheader("Sample Graph")
        
        n_nodes = st.slider("Number of nodes", 50, 500, 150, 50)
        
        # get sample of nodes
        sample_nodes = list(data.people)[:n_nodes]
        
        # get edges between them
        node_set = set(sample_nodes)
        edges = [
            {'from': h, 'relation': r, 'to': t}
            for h, r, t in data.triplets
            if h in node_set and t in node_set
        ]
        
        st.info(f"Showing {len(sample_nodes)} nodes, {len(edges)} edges")
        
        net = viz.create_graph(
            sample_nodes,
            edges,
            color_by=color_by,
            size_by=size_by,
        )
        
        net.save_graph("dashboard/temp_graph.html")
        with open("dashboard/temp_graph.html", "r") as f:
            html = f.read()
        st.components.v1.html(html, height=700, scrolling=True)
    
    # legend
    with st.expander("Color Legend"):
        if color_by == "generation":
            st.write("Generation 0 (oldest) = dark blue → Generation 6 (youngest) = red")
        elif color_by == "gender":
            st.write("Female = pink, Male = blue, Unknown = gray")
        elif color_by == "degree":
            st.write("Low degree = blue → High degree = red")



# ============ TAB: PATH FINDER ============

with tab_pathfinder:
    st.header("Path Finder")
    st.write("Find all paths between two people")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        person_a = st.text_input("Person A", value="", key="path_a")
        if person_a:
            matches_a = data.search_people(person_a, limit=5)
            if matches_a and person_a not in data.people:
                person_a = st.selectbox("Select A", matches_a, key="sel_a")
    
    with col2:
        person_b = st.text_input("Person B", value="", key="path_b")
        if person_b:
            matches_b = data.search_people(person_b, limit=5)
            if matches_b and person_b not in data.people:
                person_b = st.selectbox("Select B", matches_b, key="sel_b")
    
    with col3:
        max_hops = st.slider("Max Hops", 1, 6, 3)
    
    if st.button("Find Paths", type="primary"):
        if person_a in data.people and person_b in data.people:
            with st.spinner("Finding paths..."):
                paths = data.find_paths(person_a, person_b, max_hops)
            
            if paths:
                st.success(f"Found {len(paths)} path(s)")
                
                # show path text
                st.subheader("Relationship Chains")
                for i, path in enumerate(paths[:10]):  # limit display
                    chain = []
                    for step in path:
                        chain.append(f"{step['from']} --[{step['relation']}]--> {step['to']}")
                    st.write(f"**Path {i+1}** ({len(path)} hops):")
                    st.code(" → ".join([step['from'] for step in path] + [path[-1]['to']]))
                    with st.expander("Details"):
                        for c in chain:
                            st.write(f"  {c}")
                
                if len(paths) > 10:
                    st.info(f"Showing first 10 of {len(paths)} paths")
                
                # visualize
                st.subheader("Path Visualization")
                subgraph = data.get_path_subgraph(paths[:10])  # limit for viz
                
                # subgraph stats
                sub_stats = data.get_subgraph_stats(subgraph['nodes'])
                full_stats = data.get_full_graph_stats()
                
                stat_cols = st.columns(4)
                stat_cols[0].metric("Nodes in Path", sub_stats['num_nodes'], 
                                   f"{sub_stats['num_nodes']/full_stats['num_nodes']*100:.1f}% of graph")
                stat_cols[1].metric("Edges in Path", sub_stats['num_edges'])
                stat_cols[2].metric("Avg Degree", f"{sub_stats['avg_degree']:.1f}",
                                   f"vs {full_stats['avg_degree']:.1f} overall")
                stat_cols[3].metric("Generation Range", 
                                   f"{sub_stats['generation_range'][0]}-{sub_stats['generation_range'][1]}")
                
                net = viz.create_graph(
                    subgraph['nodes'],
                    subgraph['edges'],
                    color_by='generation',
                    size_by='degree',
                )
                
                net.save_graph("dashboard/temp_path_graph.html")
                with open("dashboard/temp_path_graph.html", "r") as f:
                    html = f.read()
                st.components.v1.html(html, height=500, scrolling=True)
                
            else:
                st.warning(f"No paths found between {person_a} and {person_b} within {max_hops} hops")
        else:
            if person_a and person_a not in data.people:
                st.error(f"Person '{person_a}' not found")
            if person_b and person_b not in data.people:
                st.error(f"Person '{person_b}' not found")


# ============ TAB: PERSON LOOKUP ============

with tab_person:
    st.header("Person Details")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        search_query = st.text_input("Search by ID", value="")
        
        if search_query:
            matches = data.search_people(search_query, limit=10)
            if matches:
                selected = st.selectbox("Matches", matches)
            else:
                st.warning("No matches")
                selected = None
        else:
            # show some default options
            selected = st.selectbox(
                "Or select from list",
                list(data.people)[:50]
            )
    
    with col2:
        if selected:
            node = data.get_node(selected)
            if node:
                st.subheader(f"Details: {selected}")
                
                # basic info
                info_cols = st.columns(4)
                info_cols[0].metric("Gender", node['gender'])
                info_cols[1].metric("Generation", node['generation'])
                info_cols[2].metric("Degree", node['degree'])
                info_cols[3].metric("Children", node['num_children'])
                
                # centrality scores
                st.markdown("---")
                st.write("**Centrality Scores:**")
                
                with st.spinner("Computing centralities..."):
                    centrality = data.get_node_centrality(selected)
                
                if centrality:
                    cent_cols = st.columns(4)
                    cent_cols[0].metric("Degree Centrality", f"{centrality.get('degree_centrality', 0):.4f}")
                    cent_cols[1].metric("Betweenness", f"{centrality.get('betweenness', 0):.4f}")
                    cent_cols[2].metric("Closeness", f"{centrality.get('closeness', 0):.4f}")
                    cent_cols[3].metric("PageRank", f"{centrality.get('pagerank', 0):.6f}")
                
                st.markdown("---")
                
                # family
                st.write("**Family:**")
                if node['mothers']:
                    st.write(f"Mothers: {', '.join(node['mothers'])}")
                if node['fathers']:
                    st.write(f"Fathers: {', '.join(node['fathers'])}")
                if node['siblings']:
                    st.write(f"Siblings: {', '.join(node['siblings'])}")
                if node['children']:
                    st.write(f"Children: {', '.join(node['children'][:10])}" + 
                            (f" ... and {len(node['children'])-10} more" if len(node['children']) > 10 else ""))
                
                # flags
                flags = []
                if node['is_founder']:
                    flags.append("🌟 Founder")
                if node['is_leaf']:
                    flags.append("🍃 Leaf")
                if node['has_anomaly']:
                    flags.append(f"⚠️ Anomalies: {node['anomalies']}")
                
                if flags:
                    st.write("**Flags:** " + ", ".join(flags))
                
                # relations grouped by type
                st.markdown("---")
                st.write("**Relations (grouped by type):**")
                
                out_rels = data.outgoing[selected]
                in_rels = data.incoming[selected]
                
                # group outgoing by relation type
                out_grouped = {}
                for r, t in out_rels:
                    if r not in out_grouped:
                        out_grouped[r] = []
                    out_grouped[r].append(t)
                
                # group incoming by relation type
                in_grouped = {}
                for r, s in in_rels:
                    if r not in in_grouped:
                        in_grouped[r] = []
                    in_grouped[r].append(s)
                
                with st.expander(f"Outgoing Relations ({len(out_rels)} total)"):
                    for rel_type, targets in sorted(out_grouped.items()):
                        st.write(f"**{rel_type}** ({len(targets)}): {', '.join(targets[:5])}" +
                                (f" ..." if len(targets) > 5 else ""))
                
                with st.expander(f"Incoming Relations ({len(in_rels)} total)"):
                    for rel_type, sources in sorted(in_grouped.items()):
                        st.write(f"**{rel_type}** ({len(sources)}): {', '.join(sources[:5])}" +
                                (f" ..." if len(sources) > 5 else ""))



with tab_stats:
    st.header("Detailed Statistics")
    
    # degree distribution
    st.subheader("Degree Distribution")
    
    degrees = [n['degree'] for n in data.node_data.values()]
    fig = px.histogram(
        degrees, nbins=30,
        title="Node Degree Distribution",
        labels={'value': 'Degree', 'count': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # high degree nodes
    st.subheader("High Degree Nodes")
    
    high_deg = data.get_high_degree_nodes(20)
    high_deg_df = pd.DataFrame(high_deg)
    st.dataframe(
        high_deg_df[['person_id', 'degree', 'generation', 'gender', 'num_children']],
        use_container_width=True
    )
    
    # anomalies
    st.subheader("Anomalies")
    
    anomalous = data.get_anomalous_nodes()
    if anomalous:
        st.write(f"Found {len(anomalous)} nodes with anomalies")
        anom_df = pd.DataFrame(anomalous)
        st.dataframe(
            anom_df[['person_id', 'anomalies', 'anomaly_severity']],
            use_container_width=True
        )
    else:
        st.success("No anomalies detected")
    
    # full data download
    st.markdown("---")
    st.subheader("Export Data")
    
    if st.button("Prepare CSV"):
        df = pd.DataFrame(data.node_data.values())
        csv = df.to_csv(index=False)
        st.download_button(
            "Download Node Features CSV",
            csv,
            "node_features.csv",
            "text/csv"
        )
    
    if st.button("Save Cache"):
        data.save_cache()
        st.success("Saved to dashboard_cache.pkl")



# ---------------- footer ----------------

st.sidebar.markdown("---")
st.sidebar.caption("MetaFAM Explorer version - idek atp i lost track")
st.sidebar.caption("Using constraint-based generation inference")

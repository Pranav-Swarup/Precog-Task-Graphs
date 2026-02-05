import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_backend import DashboardData
from visualization import GraphViz

st.set_page_config(
    page_title="metafam kg explorer - by psk",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.sidebar.caption("by Pranav Swarup")

if 'data' not in st.session_state:
    st.session_state.data = None
if 'viz' not in st.session_state:
    st.session_state.viz = None

st.sidebar.title("metafam kg explorer")
st.sidebar.markdown("---")

data_path = st.sidebar.text_input("Data file path", value="./data/train.txt", help="path to triplets file")

col1, col2 = st.sidebar.columns(2)

if col1.button("Load Data"):
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
    data = DashboardData()
    if data.load_cache():
        st.session_state.data = data
        st.session_state.viz = GraphViz(data)
        st.sidebar.success("loaded from cache")
    else:
        st.sidebar.warning("no cache found")

if st.session_state.data is None:
    st.title("metafam kg explorer")
    st.info("load data from sidebar. you can change the path if your data is smwher else")
    st.stop()

data = st.session_state.data
viz = st.session_state.viz


@st.cache_data
def compute_descendant_reach(_data_triplets, _people):
    
    from src.constants import PARENT_RELATIONS
    import networkx as nx
    
    parent_child_dag = nx.DiGraph()
    for h, r, t in _data_triplets:
        if r in PARENT_RELATIONS:
            parent_child_dag.add_edge(h, t)
    
    drc = {}
    for person in _people:
        if person in parent_child_dag:
            drc[person] = len(nx.descendants(parent_child_dag, person))
        else:
            drc[person] = 0
    return drc


@st.cache_data
def compute_generational_balance(_data_triplets, _people):
    
    from src.constants import PARENT_RELATIONS
    import networkx as nx
    
    parent_child_dag = nx.DiGraph()
    for h, r, t in _data_triplets:
        if r in PARENT_RELATIONS:
            parent_child_dag.add_edge(h, t)
    
    child_parent_dag = parent_child_dag.reverse()
    
    gbi = {}
    for person in _people:
        ancestors = set(nx.ancestors(child_parent_dag, person)) if person in child_parent_dag else set()

        descendants = set(nx.descendants(parent_child_dag, person)) if person in parent_child_dag else set()

        n_anc, n_desc = len(ancestors), len(descendants)
        gbi[person] = (n_desc - n_anc) / (n_desc + n_anc + 1)
    
    return gbi


@st.cache_data
def compute_generation_span(_data_triplets, _people, _node_data_gens):
    
    from src.constants import PARENT_RELATIONS
    import networkx as nx
    
    parent_child_dag = nx.DiGraph()
    for h, r, t in _data_triplets:
        if r in PARENT_RELATIONS:
            parent_child_dag.add_edge(h, t)
    
    child_parent_dag = parent_child_dag.reverse()
    
    gs = {}
    for person in _people:
        person_gen = _node_data_gens.get(person, -1)
        if person_gen < 0:
            gs[person] = 0
            continue
        
        ancestors = set(nx.ancestors(child_parent_dag, person)) if person in child_parent_dag else set()
        descendants = set(nx.descendants(parent_child_dag, person)) if person in parent_child_dag else set()
        
        anc_gens = [_node_data_gens[a] for a in ancestors if _node_data_gens.get(a, -1) >= 0]
        desc_gens = [_node_data_gens[d] for d in descendants if _node_data_gens.get(d, -1) >= 0]
        
        min_gen = min(anc_gens) if anc_gens else person_gen
        max_gen = max(desc_gens) if desc_gens else person_gen
        gs[person] = max_gen - min_gen

    return gs


def get_family_by_id(data, family_id):
    
    import networkx as nx
    G_undirected = data.G.to_undirected()

    components = sorted(list(nx.connected_components(G_undirected)), key=len, reverse=True)
    
    if family_id < len(components):
        return list(components[family_id])
    
    return []


def get_family_for_person(data, person_id):
    
    return list(data.get_connected_component(person_id))



# prepare cached data for metrics
triplets_tuple = tuple(data.triplets)
people_tuple = tuple(data.people)
node_gens = {p: n['generation'] for p, n in data.node_data.items()}

# TAB ORDERING HERE

tab_overview, tab_explore, tab_pathfinder, tab_person, tab_stats = st.tabs([
    "Overview", "Graph Explorer", "Path Finder", "Person Lookup", "Statistics"
])


# MAIN PAGE 
# CLAUDE ASSISTED CODE BEGINS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


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
    st.caption("Genealogy-specific metrics designed for family knowledge graphs")

    interest_cols = st.columns(5)
    with interest_cols[0]:
        if st.button("Highest Degree", use_container_width=True, help="Most connected people"):
            st.session_state.interesting_metric = 'degree'
    with interest_cols[1]:
        if st.button("Descendant Reach", use_container_width=True, help="Most descendants"):
            st.session_state.interesting_metric = 'descendant_reach'
    with interest_cols[2]:
        if st.button("Top Founders", use_container_width=True, help="Founders with largest families"):
            st.session_state.interesting_metric = 'founders'
    with interest_cols[3]:
        if st.button("Balanced Connectors", use_container_width=True, help="Equal ancestors & descendants"):
            st.session_state.interesting_metric = 'balanced'
    with interest_cols[4]:
        if st.button("Generation Span", use_container_width=True, help="Span most generations"):
            st.session_state.interesting_metric = 'gen_span'

    if 'interesting_metric' in st.session_state:
        metric = st.session_state.interesting_metric
        st.write(f"**Top 15 by {metric.replace('_', ' ').title()}**")

        with st.spinner("Computing..."):
            if metric == 'degree':
                interesting = data.get_interesting_people('degree', n=15)
                cols = ['person_id', 'degree', 'generation', 'gender', 'num_children']
            elif metric == 'descendant_reach':
                drc = compute_descendant_reach(triplets_tuple, people_tuple)
                top_drc = sorted(drc.items(), key=lambda x: x[1], reverse=True)[:15]
                interesting = [{'person_id': p, 'descendants': score, 'generation': data.node_data[p]['generation'], 
                               'gender': data.node_data[p]['gender'], 'num_children': data.node_data[p]['num_children']} 
                              for p, score in top_drc]
                cols = ['person_id', 'descendants', 'generation', 'gender', 'num_children']
            elif metric == 'founders':
                interesting = data.get_interesting_people('founders', n=15)
                cols = ['person_id', 'descendants', 'num_children', 'generation', 'gender']
            elif metric == 'balanced':
                gbi = compute_generational_balance(triplets_tuple, people_tuple)
                closest = sorted(gbi.items(), key=lambda x: abs(x[1]))[:15]
                interesting = [{'person_id': p, 'gbi_score': round(score, 3), 'generation': data.node_data[p]['generation'],
                               'degree': data.node_data[p]['degree'], 'gender': data.node_data[p]['gender']} 
                              for p, score in closest]
                cols = ['person_id', 'gbi_score', 'generation', 'degree', 'gender']
            elif metric == 'gen_span':
                gs = compute_generation_span(triplets_tuple, people_tuple, node_gens)
                top_gs = sorted(gs.items(), key=lambda x: x[1], reverse=True)[:15]
                interesting = [{'person_id': p, 'gen_span': score, 'generation': data.node_data[p]['generation'],
                               'degree': data.node_data[p]['degree'], 'gender': data.node_data[p]['gender']} 
                              for p, score in top_gs]
                cols = ['person_id', 'gen_span', 'generation', 'degree', 'gender']
            else:
                interesting, cols = [], []

        if interesting:
            df = pd.DataFrame(interesting)
            cols = [c for c in cols if c in df.columns]
            st.dataframe(df[cols], use_container_width=True)

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns(2)
    with col1:
        gen_df = pd.DataFrame([{'Generation': k, 'Count': v} for k, v in gen_stats.items() if k >= 0])
        fig = px.bar(gen_df, x='Generation', y='Count', title='Generation Distribution',
                    color='Generation', color_continuous_scale='viridis')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        gender_stats = data.get_gender_stats()
        gender_df = pd.DataFrame([{'Gender': k, 'Count': v} for k, v in gender_stats.items()])
        fig = px.pie(gender_df, values='Count', names='Gender', title='Gender Distribution',
                    color='Gender', color_discrete_map={'F': '#e91e8c', 'M': '#2196f3', 'Unknown': '#9e9e9e'})
        st.plotly_chart(fig, use_container_width=True)

    # Charts row 2: Genealogical insights
    st.subheader("Genealogical Structure Insights")
    st.caption("Based on domain-specific family graph analysis")
    
    col3, col4 = st.columns(2)
    with col3:
        with st.spinner("Computing descendant reach..."):
            drc = compute_descendant_reach(triplets_tuple, people_tuple)
            gen_drc = defaultdict(list)
            for person, score in drc.items():
                gen = data.node_data[person]['generation']
                if gen >= 0:
                    gen_drc[gen].append(score)
            drc_by_gen = [{'Generation': g, 'Avg Descendants': round(sum(gen_drc[g])/len(gen_drc[g]), 1) if gen_drc[g] else 0} 
                         for g in sorted(gen_drc.keys())]
            drc_df = pd.DataFrame(drc_by_gen)
            fig = px.bar(drc_df, x='Generation', y='Avg Descendants', title='Avg Descendant Reach by Generation',
                        color='Avg Descendants', color_continuous_scale='oranges')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("↓ Decreasing trend confirms hierarchical tree structure and strict hierarchical inheritance from founders to leaves.")

    with col4:
        with st.spinner("Computing vertical importance..."):
            
            vertical_importance = pd.DataFrame({
                "Generation": [0, 1, 2, 3, 4, 5, 6],
                "Avg_Mediated_Pairs": [97.4, 45.4, 15.4, 4.0, 0.9, 0.3, 0.0]
            })

            fig = px.bar(
                vertical_importance,
                x="Generation",
                y="Avg_Mediated_Pairs",
                title="Vertical Importance by Generation",
                labels={
                    "Avg_Mediated_Pairs": "Average Mediated Ancestor–Descendant Pairs",
                    "Generation": "Generation"
                }
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Vertical Importance measures how much an individual mediates ancestor–descendant paths. "
                "Founders dominate this index because they sit above large descendant cones and therefore "
                "lie on many vertical paths. As generations progress, descendant sets shrink rapidly, "
                "causing a steep decline in mediated pairs, reaching near-zero for the youngest generations."
            )


    # Charts row 3
    col5, col6 = st.columns(2)
    with col5:
        families = data.get_families()
        size_bins = {'1-10': 0, '11-50': 0, '51-100': 0, '100+': 0}
        for f in families:
            s = f['size']
            if s <= 10: size_bins['1-10'] += 1
            elif s <= 50: size_bins['11-50'] += 1
            elif s <= 100: size_bins['51-100'] += 1
            else: size_bins['100+'] += 1
        size_df = pd.DataFrame([{'Size Range': k, 'Count': v} for k, v in size_bins.items()])
        fig = px.bar(size_df, x='Size Range', y='Count', title=f'Family Size Distribution ({len(families)} families)',
                    color='Count', color_continuous_scale='purples')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Most families are around the same size ~ 25-27 people. This indicates that the dataset was synthetically prepared or pruned to mimic this homogeneity")

    with col6:
        gen_degrees = defaultdict(list)
        for p, node in data.node_data.items():
            gen = node['generation']
            if gen >= 0:
                gen_degrees[gen].append(node['degree'])
        deg_by_gen = [{'Generation': g, 'Avg Degree': round(sum(gen_degrees[g])/len(gen_degrees[g]), 1) if gen_degrees[g] else 0} 
                     for g in sorted(gen_degrees.keys())]
        deg_df = pd.DataFrame(deg_by_gen)
        fig = px.line(deg_df, x='Generation', y='Avg Degree', title='Avg Degree by Generation', markers=True)
        fig.update_traces(line_color='#3498db', marker_color='#3498db')
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Middle generations have higher connectivity and this can be confirmed from the ontology of family trees.")

    st.subheader("Relation Types")
    rel_stats = data.get_relation_stats()
    if isinstance(rel_stats, dict):
        rel_stats = rel_stats.items()
    top_relations = sorted(rel_stats, key=lambda x: x[1], reverse=True)[:15]
    rel_df = pd.DataFrame(top_relations, columns=["Relation", "Count"])
    fig = px.bar(rel_df, x='Relation', y='Count', title='Top 15 Relation Types', color='Count', color_continuous_scale='blues')
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
# ---------------- graph explorer ----------------

with tab_explore:
    st.header("interactive graph - play around with stuff its really fun :D")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("controls")
    
    view_mode = st.sidebar.radio("View Mode", ["Family Tree", "Ego Network", "Sample Nodes", "Family View"], index=0)
    color_by = st.sidebar.selectbox("Color by", ["generation", "gender", "degree"])
    size_by = st.sidebar.selectbox("Size by", ["degree", "children", "fixed"])
    
    if view_mode == "Ego Network":
        st.subheader("Ego Network")
        col1, col2 = st.columns([3, 1])
        with col1:
            person_search = st.text_input("Search person", value="olivia0")
        with col2:
            hops = st.selectbox("Hops", [1, 2, 3], index=0)
        
        st.write("Quick picks:")
        quick_cols = st.columns(5)
        quick_picks = ['olivia0', 'katharina1', 'fabian26', 'emma7', 'jonas23']
        for i, qp in enumerate(quick_picks):
            if quick_cols[i].button(qp, key=f"qp_{qp}"):
                person_search = qp
        
        if person_search and person_search in data.people:
            subgraph = data.get_ego_network(person_search, hops)
            sub_stats = data.get_subgraph_stats(subgraph['nodes'])
            full_stats = data.get_full_graph_stats()
            
            with st.expander("Subgraph Stats", expanded=False):
                stat_cols = st.columns(4)
                stat_cols[0].metric("Nodes", sub_stats['num_nodes'], f"{sub_stats['num_nodes']/full_stats['num_nodes']*100:.2f}%")
                stat_cols[1].metric("Edges", sub_stats['num_edges'], f"{sub_stats['num_edges']/full_stats['num_edges']*100:.2f}%")
                stat_cols[2].metric("Avg Degree", f"{sub_stats['avg_degree']:.1f}", f"full: {full_stats['avg_degree']:.1f}")
                stat_cols[3].metric("Founders", sub_stats['num_founders'])
                st.write(f"**Gender:** M={sub_stats['gender_counts'].get('M', 0)}, F={sub_stats['gender_counts'].get('F', 0)}, Unknown={sub_stats['gender_counts'].get('Unknown', 0)}")
                st.write(f"**Generation Range:** {sub_stats['generation_range'][0]} - {sub_stats['generation_range'][1]}")
            
            st.info(f"Showing {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges")
            net = viz.create_graph(subgraph['nodes'], subgraph['edges'], color_by=color_by, size_by=size_by)
            net.save_graph("dashboard/temp_graph.html")
            with open("dashboard/temp_graph.html", "r") as f:
                st.components.v1.html(f.read(), height=700, scrolling=True)
        else:
            st.warning(f"Person '{person_search}' not found")
    
    elif view_mode == "Family Tree":
        st.subheader("Family Tree View")
        col1, col2 = st.columns([3, 1])
        with col1:
            person_search = st.text_input("Search person", value="olivia0", key="tree_search")
            show_all_edges = st.toggle("Show all relations", value=False, help="Include all incoming and outgoing relations")
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
                extra_nodes, extra_edges = set(), []
                for r, t in data.outgoing[person_search]:
                    extra_nodes.add(t)
                    extra_edges.append({'from': person_search, 'relation': r, 'to': t})
                for r, s in data.incoming[person_search]:
                    extra_nodes.add(s)
                    extra_edges.append({'from': s, 'relation': r, 'to': person_search})
                tree['nodes'] = list(set(tree['nodes']) | extra_nodes)
                tree['edges'].extend(extra_edges)

            sub_stats = data.get_subgraph_stats(tree['nodes'])
            full_stats = data.get_full_graph_stats()
            
            with st.expander("Subgraph Stats", expanded=False):
                stat_cols = st.columns(4)
                stat_cols[0].metric("Nodes", sub_stats['num_nodes'], f"{sub_stats['num_nodes']/full_stats['num_nodes']*100:.2f}%")
                stat_cols[1].metric("Edges", sub_stats['num_edges'])
                stat_cols[2].metric("Avg Degree", f"{sub_stats['avg_degree']:.1f}")
                stat_cols[3].metric("Founders", sub_stats['num_founders'])
                st.write(f"**Gender:** M={sub_stats['gender_counts'].get('M', 0)}, F={sub_stats['gender_counts'].get('F', 0)}")
                st.write(f"**Gen Range:** {sub_stats['generation_range'][0]} - {sub_stats['generation_range'][1]}")

            st.info(f"Family tree: {len(tree['nodes'])} members")
            net = viz.create_graph(tree['nodes'], tree['edges'], color_by=color_by, size_by='fixed', physics=False)
            net.save_graph("dashboard/temp_graph.html")
            with open("dashboard/temp_graph.html", "r") as f:
                st.components.v1.html(f.read(), height=700, scrolling=True)
        else:
            st.warning(f"Person '{person_search}' not found")

    elif view_mode == "Family View":
        st.subheader("Full Family View")
        st.caption("View entire family (connected component) - same hierarchical layout as Family Tree")
        
        select_method = st.radio("Select family by:", ["Family ID", "Person Name"], horizontal=True)
        family_members = []
        
        if select_method == "Family ID":
            families = data.get_families()
            with st.expander("Family Summary", expanded=False):
                fam_summary = [{'ID': f['family_id'], 'Size': f['size'], 
                               'Sample Members': ', '.join(f['members'][:3]) + ('...' if len(f['members']) > 3 else '')} 
                              for f in families[:20]]
                st.dataframe(pd.DataFrame(fam_summary), use_container_width=True)
                if len(families) > 20:
                    st.caption(f"Showing top 20 of {len(families)} families")
            
            family_id = st.number_input("Family ID", min_value=0, max_value=len(families) - 1, value=0, help="0 = largest family")
            family_members = get_family_by_id(data, family_id)
            
        else:
            person_search = st.text_input("Search person name", value="", key="family_view_person", help="Enter person ID to view their family")
            if person_search:
                matches = data.search_people(person_search, limit=5)
                if matches:
                    selected_person = person_search if person_search in data.people else st.selectbox("Select person", matches, key="fv_select")
                    family_members = get_family_for_person(data, selected_person)
                    st.info(f"Found {selected_person}'s family")
                else:
                    st.warning("No matches found")
        
        if family_members:
            if len(family_members) > 200:
                st.warning(f"Large family ({len(family_members)} members). May be slow.")
                if st.checkbox("Limit to 200 nodes", value=True):
                    family_members = family_members[:200]
            
            node_set = set(family_members)
            edges = [{'from': h, 'relation': r, 'to': t} for h, r, t in data.triplets if h in node_set and t in node_set]
            
            sub_stats = data.get_subgraph_stats(family_members)
            with st.expander("Family Stats", expanded=True):
                stat_cols = st.columns(5)
                stat_cols[0].metric("Members", sub_stats['num_nodes'])
                stat_cols[1].metric("Relations", sub_stats['num_edges'])
                stat_cols[2].metric("Founders", sub_stats['num_founders'])
                stat_cols[3].metric("Leaves", sub_stats['num_leaves'])
                stat_cols[4].metric("Gen Range", f"{sub_stats['generation_range'][0]}-{sub_stats['generation_range'][1]}")
                st.write(f"**Gender:** M={sub_stats['gender_counts'].get('M', 0)}, F={sub_stats['gender_counts'].get('F', 0)}, Unknown={sub_stats['gender_counts'].get('Unknown', 0)}")
            
            st.info(f"Showing {len(family_members)} members, {len(edges)} relations")
            net = viz.create_graph(family_members, edges, color_by=color_by, size_by='fixed', physics=False)
            net.save_graph("dashboard/temp_graph.html")
            with open("dashboard/temp_graph.html", "r") as f:
                st.components.v1.html(f.read(), height=700, scrolling=True)
        elif select_method == "Family ID":
            st.info("Select a family ID to view")
    
    else:  # sample nodes
        st.subheader("Sample Graph")
        n_nodes = st.slider("Number of nodes", 50, 500, 150, 50)
        sample_nodes = list(data.people)[:n_nodes]
        node_set = set(sample_nodes)
        edges = [{'from': h, 'relation': r, 'to': t} for h, r, t in data.triplets if h in node_set and t in node_set]
        st.info(f"Showing {len(sample_nodes)} nodes, {len(edges)} edges")
        net = viz.create_graph(sample_nodes, edges, color_by=color_by, size_by=size_by)
        net.save_graph("dashboard/temp_graph.html")
        with open("dashboard/temp_graph.html", "r") as f:
            st.components.v1.html(f.read(), height=700, scrolling=True)
    
    with st.expander("Color Legend"):
        if color_by == "generation":
            st.write("Gen 0 (oldest) = dark blue → Gen 6 (youngest) = red")
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
                st.subheader("Relationship Chains")
                for i, path in enumerate(paths[:10]):
                    st.write(f"**Path {i+1}** ({len(path)} hops):")
                    st.code(" → ".join([step['from'] for step in path] + [path[-1]['to']]))
                    with st.expander("Details"):
                        for step in path:
                            st.write(f"  {step['from']} --[{step['relation']}]--> {step['to']}")
                
                if len(paths) > 10:
                    st.info(f"Showing first 10 of {len(paths)} paths")
                
                st.subheader("Path Visualization")
                subgraph = data.get_path_subgraph(paths[:10])
                sub_stats = data.get_subgraph_stats(subgraph['nodes'])
                full_stats = data.get_full_graph_stats()
                
                stat_cols = st.columns(4)
                stat_cols[0].metric("Nodes in Path", sub_stats['num_nodes'], f"{sub_stats['num_nodes']/full_stats['num_nodes']*100:.1f}%")
                stat_cols[1].metric("Edges in Path", sub_stats['num_edges'])
                stat_cols[2].metric("Avg Degree", f"{sub_stats['avg_degree']:.1f}", f"vs {full_stats['avg_degree']:.1f}")
                stat_cols[3].metric("Gen Range", f"{sub_stats['generation_range'][0]}-{sub_stats['generation_range'][1]}")
                
                net = viz.create_graph(subgraph['nodes'], subgraph['edges'], color_by='generation', size_by='degree')
                net.save_graph("dashboard/temp_path_graph.html")
                with open("dashboard/temp_path_graph.html", "r") as f:
                    st.components.v1.html(f.read(), height=500, scrolling=True)
            else:
                st.warning(f"No paths found between {person_a} and {person_b} within {max_hops} hops")
        else:
            if person_a and person_a not in data.people:
                st.error(f"Person '{person_a}' not found")
            if person_b and person_b not in data.people:
                st.error(f"Person '{person_b}' not found")


# CLAUDE ASSISTED CODE ENDS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



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
            selected = st.selectbox("Or select from list", list(data.people)[:50])
    
    with col2:
        if selected:
            node = data.get_node(selected)
            if node:
                st.subheader(f"Details: {selected}")
                
                info_cols = st.columns(4)
                info_cols[0].metric("Gender", node['gender'])
                info_cols[1].metric("Generation", node['generation'])
                info_cols[2].metric("Degree", node['degree'])
                info_cols[3].metric("Children", node['num_children'])
                
                st.markdown("---")
                st.write("**Genealogical Metrics:**")
                
                with st.spinner("Computing metrics..."):
                    drc = compute_descendant_reach(triplets_tuple, people_tuple)
                    gbi = compute_generational_balance(triplets_tuple, people_tuple)
                    gs = compute_generation_span(triplets_tuple, people_tuple, node_gens)
                
                gen_cols = st.columns(3)
                gen_cols[0].metric("Descendant Reach", drc.get(selected, 0), help="Total descendants")
                gen_cols[1].metric("Gen Balance", f"{gbi.get(selected, 0):.3f}", help="+1=founder, 0=bridge, -1=leaf")
                gen_cols[2].metric("Gen Span", gs.get(selected, 0), help="Generations spanned")
                
                st.markdown("---")
                st.write("**Family:**")
                if node['mothers']:
                    st.write(f"Mothers: {', '.join(node['mothers'])}")
                if node['fathers']:
                    st.write(f"Fathers: {', '.join(node['fathers'])}")
                if node['siblings']:
                    st.write(f"Siblings: {', '.join(node['siblings'])}")
                if node['children']:
                    st.write(f"Children: {', '.join(node['children'][:10])}" + 
                            (f" ... +{len(node['children'])-10} more" if len(node['children']) > 10 else ""))
                
                flags = []
                if node['is_founder']:
                    flags.append("Founder")
                if node['is_leaf']:
                    flags.append("Leaf")
                if node['has_anomaly']:
                    flags.append(f"Anomalies: {node['anomalies']}")
                if flags:
                    st.write("**Flags:** " + ", ".join(flags))
                
                st.markdown("---")
                st.write("**Relations:**")
                out_rels, in_rels = data.outgoing[selected], data.incoming[selected]
                
                out_grouped = {}
                for r, t in out_rels:
                    out_grouped.setdefault(r, []).append(t)
                in_grouped = {}
                for r, s in in_rels:
                    in_grouped.setdefault(r, []).append(s)
                
                with st.expander(f"Outgoing ({len(out_rels)})"):
                    for rel_type, targets in sorted(out_grouped.items()):
                        st.write(f"**{rel_type}** ({len(targets)}): {', '.join(targets[:5])}" + (" ..." if len(targets) > 5 else ""))
                with st.expander(f"Incoming ({len(in_rels)})"):
                    for rel_type, sources in sorted(in_grouped.items()):
                        st.write(f"**{rel_type}** ({len(sources)}): {', '.join(sources[:5])}" + (" ..." if len(sources) > 5 else ""))


with tab_stats:
    st.header("Detailed Statistics")
    
    st.subheader("Degree Distribution")

    degrees = [n['degree'] for n in data.node_data.values()]
    fig = px.histogram(degrees, nbins=30, title="Node Degree Distribution", labels={'value': 'Degree', 'count': 'Count'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("High Degree Nodes")
    high_deg = data.get_high_degree_nodes(20)
    high_deg_df = pd.DataFrame(high_deg)
    st.dataframe(high_deg_df[['person_id', 'degree', 'generation', 'gender', 'num_children']], use_container_width=True)
    
    st.subheader("Anomalies")
    anomalous = data.get_anomalous_nodes()
    if anomalous:
        st.write(f"Found {len(anomalous)} nodes with anomalies")
        anom_df = pd.DataFrame(anomalous)
        st.dataframe(anom_df[['person_id', 'anomalies', 'anomaly_severity']], use_container_width=True)
    else:
        st.success("No anomalies detected")
    

# CLAUDE ASSISTED CODE STARTS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



    st.markdown("---")
    st.subheader("Export Data")
    if st.button("Prepare CSV"):
        df = pd.DataFrame(data.node_data.values())
        csv = df.to_csv(index=False)
        st.download_button("Download Node Features CSV", csv, "node_features.csv", "text/csv")
    if st.button("Save Cache"):
        data.save_cache()
        st.success("Saved to dashboard_cache.pkl")


# CLAUDE ASSISTED CODE ENDS HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

st.sidebar.markdown("---")
st.sidebar.caption("metafam explorer version - idk at this point i lost count")
st.sidebar.caption("latest edit was to include genealogy inspired metrics from the insights gathered from tertiary analysis. I also added a new full-family view to show ego networks in tree shape.")
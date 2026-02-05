from pyvis.network import Network
from typing import Dict, List, Optional, Set


# color schemes
GENERATION_COLORS = {
    -1: '#888888',  # unknown
    0: '#1a5276',   # dark blue - oldest
    1: '#2874a6',
    2: '#3498db',
    3: '#52be80',   # green - middle
    4: '#f4d03f',   # yellow
    5: '#e67e22',   # orange
    6: '#e74c3c',   # red - youngest
}

GENDER_COLORS = {
    'F': '#e91e8c',
    'M': '#2196f3',
    'Unknown': '#9e9e9e',
    'Ambiguous': '#ff9800',
}

RELATION_COLORS = {
    'vertical': '#3498db',    # blue for parent-child
    'horizontal': '#2ecc71',  # green for siblings/cousins
    'unknown': '#95a5a6',
}

# specific relation colors
EDGE_COLORS = {
    # parent relations - green
    'motherOf': '#2ecc71',
    'fatherOf': '#2ecc71',
    'parentOf': '#2ecc71',
    # brother - blue
    'brotherOf': '#2196f3',
    # sister - pink
    'sisterOf': '#e91e8c',
    # sibling generic
    'siblingOf': '#5dade2',
    # cousin relations - shades of purple
    'cousinOf': '#9b59b6',
    'maternalCousinOf': '#8e44ad',
    'paternalCousinOf': '#a569bd',
    # nephew/niece - lighter purple
    'nephewOf': '#bb8fce',
    'nieceOf': '#d2b4de',
    'uncleOf': '#7d3c98',
    'auntOf': '#af7ac5',
    # husband/wife - red
    'husbandOf': '#e74c3c',
    'wifeOf': '#e74c3c',
    'marriedTo': '#e74c3c',
    'spouseOf': '#e74c3c',
}


class GraphViz:
    """
    creates pyvis graphs from dashboard data
    """
    
    def __init__(self, data_backend):
        self.data = data_backend
    
    def create_graph(
        self,
        nodes: List[str],
        edges: List[dict],
        color_by: str = 'generation',
        size_by: str = 'degree',
        height: str = '700px',
        physics: bool = True,
    ) -> Network:
        """
        create pyvis network from node/edge lists
        
        nodes: list of person_ids
        edges: list of {'from': x, 'relation': r, 'to': y}
        color_by: 'generation', 'gender', 'degree'
        size_by: 'degree', 'children', 'fixed'
        """
        
        net = Network(
            height=height,
            width='100%',
            directed=True,
            notebook=False,
            bgcolor='#ffffff',
            font_color='#333333',
        )
        
        # physics settings
        if physics:
            net.set_options('''
            {
                "physics": { "enabled": false },
                "layout": {
                    "improvedLayout": true
                },
                "nodes": {
                    "font": { "size": 14, "face": "arial" }
                },
                "edges": {
                    "smooth": { "type": "cubicBezier" },
                    "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } }
                },
                "interaction": {
                    "dragNodes": true,
                    "dragView": true,
                    "zoomView": true
                }
            }
            ''')

        else:
            # hierarchical layout for tree views - older gens at top
            net.set_options('''
            {
                "physics": {"enabled": false},
                "layout": {
                    "hierarchical": {
                        "enabled": false
                    }
                },
                "nodes": {
                    "font": {"size": 14, "face": "arial"}
                },
                "edges": {
                    "smooth": {"type": "cubicBezier"},
                    "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
                },
                "interaction": {
                    "dragNodes": true,
                    "dragView": true,
                    "zoomView": true
                }
            }
            ''')


        # add nodes
        for person_id in nodes:
            node_info = self.data.get_node(person_id)
            if node_info is None:
                continue

            gen = node_info.get('generation', -1)
            
            color = self._get_color(node_info, color_by)
            size = self._get_size(node_info, size_by)
            shape = self._get_shape(node_info)
            label = self._get_label(node_info)
            title = self._get_title(node_info)
            
            net.add_node(
                person_id,
                label=label,
                title=title,
                color=color,
                size=size,
                shape=shape,
                level=gen if gen >= 0 else None 
            )

        if physics:
            import math

            n = len(net.nodes)
            if n > 1:
                R = 1000  # increase for more spread
                angle_step = 2 * math.pi / n

                for i, node in enumerate(net.nodes):
                    angle = i * angle_step
                    node['x'] = int(R * math.cos(angle))
                    node['y'] = int(R * math.sin(angle))
                    node['fixed'] = False
        else:
            # manual hierarchical positioning for family tree
            # group nodes by generation and lay out in rows
            from collections import defaultdict
            gen_groups = defaultdict(list)
            for node in net.nodes:
                gen = node.get('level', -1) if node.get('level') is not None else -1
                gen_groups[gen].append(node)
            
            y_spacing = 150
            x_spacing = 200
            
            for gen, gen_nodes in gen_groups.items():
                y_pos = gen * y_spacing
                n_in_gen = len(gen_nodes)
                start_x = -(n_in_gen - 1) * x_spacing / 2
                
                for i, node in enumerate(gen_nodes):
                    node['x'] = int(start_x + i * x_spacing)
                    node['y'] = int(y_pos)
                    node['fixed'] = False
        
        # add edges
        node_set = set(nodes)
        for edge in edges:
            h, r, t = edge['from'], edge['relation'], edge['to']
            if h not in node_set or t not in node_set:
                continue
            
            edge_color = self._get_edge_color(r)
            
            net.add_edge(
                h, t,
                title=r,
                color=edge_color,
                width=1.5,
            )
        
        return net
    
    def _get_color(self, node: dict, color_by: str) -> str:
        if color_by == 'generation':
            gen = node.get('generation', -1)
            return GENERATION_COLORS.get(gen, GENERATION_COLORS.get(6))  # cap at 6
        elif color_by == 'gender':
            return GENDER_COLORS.get(node.get('gender', 'Unknown'))
        elif color_by == 'degree':
            deg = node.get('degree', 0)
            if deg < 10:
                return '#3498db'
            elif deg < 20:
                return '#f39c12'
            elif deg < 30:
                return '#e67e22'
            else:
                return '#c0392b'
        return '#3498db'
    
    def _get_size(self, node: dict, size_by: str) -> int:
        if size_by == 'degree':
            deg = node.get('degree', 1)
            return max(15, min(50, 10 + deg))
        elif size_by == 'children':
            children = node.get('num_children', 0)
            return max(15, min(50, 15 + children * 5))
        return 20
    
    def _get_shape(self, node: dict) -> str:

        if node.get('has_anomaly'):
            return 'triangle'

        gender = node.get('gender', 'Unknown')
        if gender == 'M':
            return 'square'
        elif gender == 'F':
            return 'dot'
        return 'dot'
    
    def _get_label(self, node: dict) -> str:
        pid = node['person_id']
        gen = node.get('generation', '?')
        return f"{pid}\n(g{gen})"
    
    def _get_title(self, node: dict) -> str:
        """hover text - plain text, no html"""
        lines = [
            f"{node['person_id']}",
            f"Gender: {node.get('gender', '?')}",
            f"Generation: {node.get('generation', '?')}",
            f"Degree: {node.get('degree', 0)}",
            f"Parents: {node.get('num_parents', 0)}",
            f"Children: {node.get('num_children', 0)}",
            f"Siblings: {node.get('num_siblings', 0)}",
        ]
        
        if node.get('is_founder'):
            lines.append("[Founder]")
        if node.get('is_leaf'):
            lines.append("[Leaf]")
        if node.get('has_anomaly'):
            lines.append(f"Anomalies: {', '.join(node.get('anomalies', []))}")
        
        return '\n'.join(lines)
    
    def _get_edge_color(self, relation: str) -> str:
        # check specific relation first
        if relation in EDGE_COLORS:
            return EDGE_COLORS[relation]
        # fall back to category
        cat = self.data.categorize_relation(relation)
        return RELATION_COLORS.get(cat, RELATION_COLORS['unknown'])
    
    def create_ego_graph(
        self,
        person_id: str,
        hops: int = 1,
        color_by: str = 'generation',
        size_by: str = 'degree',
    ) -> Network:
        
        
        subgraph = self.data.get_ego_network(person_id, hops)
        return self.create_graph(
            subgraph['nodes'],
            subgraph['edges'],
            color_by=color_by,
            size_by=size_by,
            #physics=False,  # physics messed up a lot here. kept juggling
        )
    
    def create_family_tree(
        self,
        person_id: str,
        color_by: str = 'generation',
    ) -> Network:
        
        tree = self.data.get_family_tree(person_id)
        return self.create_graph(
            tree['nodes'],
            tree['edges'],
            color_by=color_by,
            size_by='fixed'  # trees look better without physics
        )
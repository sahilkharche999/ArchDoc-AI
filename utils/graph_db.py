from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()

class ConstructionGraph:
    def __init__(self):
        # Connection Details
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "password")
        
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # =========================================================================
    # 1. AGENT 1: TEXT RULES (The Definitions)
    # =========================================================================
    
    def add_schedule_rule(self, schedule_name, symbol, specs, page_num):
        """
        Stores a rule from a schedule.
        Example: Schedule="Shear Wall", Symbol="<1>", Specs="5/8 bolt @ 16oc"
        """
        query = """
        MERGE (s:Schedule {name: $schedule_name})
        MERGE (p:Page {number: $page_num})
        MERGE (s)-[:ON_PAGE]->(p)
        
        MERGE (r:Rule {symbol: $symbol})
        SET r.specs = $specs
        
        MERGE (s)-[:DEFINES]->(r)
        """
        with self.driver.session() as session:
            session.run(query, schedule_name=schedule_name, symbol=symbol, specs=specs, page_num=page_num)
            print(f"Graph: Added Rule {symbol} to {schedule_name}")

    # =========================================================================
    # 2. AGENT 3: DETAIL BOM (The Recipes)
    # =========================================================================

    def add_detail_bom(self, detail_key, title, materials_list, page_num):
        """
        Stores a Detail and its Bill of Materials.
        Example: Key="7/S-3.2", Materials=[{name: "MC6x15.1", qty: "Variable"}]
        """
        query = """
        MERGE (d:Detail {key: $detail_key})
        SET d.title = $title
        MERGE (p:Page {number: $page_num})
        MERGE (d)-[:ON_PAGE]->(p)
        
        FOREACH (mat IN $materials |
            MERGE (m:Material {name: mat.item_name})
            MERGE (d)-[:REQUIRES {qty_rule: mat.qty_rule, notes: mat.notes}]->(m)
        )
        """
        # Clean list for Cypher (Pydantic to Dict)
        clean_materials = [
            {"item_name": m["item_name"], "qty_rule": m["qty_rule"], "notes": m.get("notes", "")} 
            for m in materials_list
        ]
        
        with self.driver.session() as session:
            session.run(query, detail_key=detail_key, title=title, materials=clean_materials, page_num=page_num)
            print(f"Graph: Added BOM for {detail_key}")

    # =========================================================================
    # 3. AGENT 2: PLAN INSTANCES (The Map)
    # =========================================================================

    def add_plan_instance(self, item_type, label, location, associated_text, page_num):
        """
        Stores an item found on the floor plan.
        Crucially, it tries to LINK to a Rule or Detail immediately.
        """
        query = """
        CREATE (i:Instance {type: $item_type, label: $label})
        SET i.location = $location
        SET i.dimension = $associated_text
        
        MERGE (p:Page {number: $page_num})
        MERGE (i)-[:LOCATED_ON]->(p)
        
        // AUTOMATIC LINKING LOGIC (Connecting the Dots)
        
        // 1. Link to Detail (e.g. Label="7/S-3.2" matches Detail Key)
        WITH i
        OPTIONAL MATCH (d:Detail {key: i.label})
        FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
            MERGE (i)-[:IS_INSTANCE_OF]->(d)
        )

        // 2. Link to Rule (e.g. Label="<1>" matches Rule Symbol)
        WITH i
        OPTIONAL MATCH (r:Rule {symbol: i.label})
        FOREACH (_ IN CASE WHEN r IS NOT NULL THEN [1] ELSE [] END |
            MERGE (i)-[:FOLLOWS_RULE]->(r)
        )
        """
        with self.driver.session() as session:
            session.run(query, item_type=item_type, label=label, location=location, associated_text=associated_text, page_num=page_num)
            # print(f"Graph: Added Instance {label} at {location}")

    # =========================================================================
    # 4. AGENT 4: THE MERGER QUERY (The Payoff)
    # =========================================================================

    def get_full_estimation_data(self):
        """
        Retrieves all Instances and their connected Definitions.
        This is the 'Magic Query' that gives Agent 4 everything it needs.
        """
        query = """
        MATCH (i:Instance)
        
        // Get Detail Info if linked
        OPTIONAL MATCH (i)-[:IS_INSTANCE_OF]->(d:Detail)-[req:REQUIRES]->(m:Material)
        
        // Get Rule Info if linked
        OPTIONAL MATCH (i)-[:FOLLOWS_RULE]->(r:Rule)<-[:DEFINES]-(s:Schedule)
        
        RETURN 
            i.type as Type,
            i.label as Label,
            i.location as Location,
            i.dimension as Dimension,
            d.key as Detail_Key,
            collect({mat: m.name, rule: req.qty_rule}) as Detail_BOM,
            r.specs as Rule_Specs,
            s.name as Schedule_Name
        """
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

# Singleton
graph_db = ConstructionGraph()
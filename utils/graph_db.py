from neo4j import GraphDatabase
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

class ConstructionGraph:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Initialize Embedding Model
        self.embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
        
        # Create Vector Index (Run this once)
        self.create_vector_index()

    def close(self):
        self.driver.close()

    
    def create_vector_index(self):
        """Creates a Vector Index on Definition nodes if it doesn't exist."""
        query = """
        CREATE VECTOR INDEX definition_index IF NOT EXISTS
        FOR (d:Definition)
        ON (d.embedding)
        OPTIONS {indexConfig: {
         `vector.dimensions`: 3072,  
         `vector.similarity_function`: 'cosine'
        }}
        """
        with self.driver.session() as session:
            session.run(query)

    # --- INGESTION (With Embeddings) ---

    def add_schedule_rule(self, project_id, schedule_name, symbol, specs, page_num):
        print(f"DEBUG: Adding Rule {symbol} to {schedule_name}...") # <--- ADD THIS
        """
        Stores a rule from a schedule with Vector Embeddings.
        """
        # 1. Create Description for Embedding
        # Example: "Symbol: <1>. Schedule: Shear Wall. Specs: 5/8 bolt @ 16oc"
        try:
            # 1. Create Description
            description = f"Symbol: {symbol}. Schedule: {schedule_name}. Specs: {specs}"
            
            # 2. Generate Vector
            print("DEBUG: Generating Vector...") # <--- ADD THIS
            vector = self.embedder.embed_query(description)
            print(f"DEBUG: Vector Generated (Size: {len(vector)})") # <--- ADD THIS
            
        
            # 3. Cypher Query
            query = """
            MERGE (proj:Project {id: $project_id})
            MERGE (p:Sheet {number: $page_num, project: $project_id})
            MERGE (p)-[:BELONGS_TO]->(proj)
            
            // Create/Update Definition Node
            MERGE (d:Definition {id: $symbol, project: $project_id})
            SET d:Schedule
            SET d.name = $schedule_name
            SET d.specs = $specs
            
            // GraphRAG Fields
            SET d.text = $description
            SET d.embedding = $vector
            
            MERGE (d)-[:FOUND_ON]->(p)
            """
        
        # 4. Execute
            with self.driver.session() as session:
                session.run(
                    query, 
                    project_id=project_id, 
                    schedule_name=schedule_name, 
                    symbol=symbol, 
                    specs=specs, 
                    page_num=page_num,
                    description=description,
                    vector=vector
                )
                print(f"Graph: Added Rule '{symbol}' with Vector.")
        except Exception as e:
            print(f"CRITICAL GRAPH ERROR: {e}") # <--- CATCH ERRORS

    def add_detail_bom(self, project_id, detail_key, title, materials_list, page_num):
        """
        Stores a Detail and its BOM, including Vector Embeddings for GraphRAG.
        """
        # 1. Create rich description for embedding
        # Example: "Detail: 7/S-3.2. Title: Ladder. Contains: MC6x15.1, L4x4"
        mat_text = ", ".join([m["item_name"] for m in materials_list])
        description = f"Detail: {detail_key}. Title: {title}. Contains: {mat_text}"
        
        # 2. Generate Vector
        vector = self.embedder.embed_query(description)
        
        # 3. Prepare Data for Cypher (Convert Pydantic/Dict to clean list)
        clean_materials = [
            {"item_name": m["item_name"], "qty_rule": m["qty_rule"], "notes": m.get("notes", "")} 
            for m in materials_list
        ]
        
        # 4. The Cypher Query
        query = """
        MERGE (proj:Project {id: $project_id})
        MERGE (p:Sheet {number: $page_num, project: $project_id})
        MERGE (p)-[:BELONGS_TO]->(proj)
        
        // Create/Update the Definition Node
        MERGE (d:Definition {id: $detail_key, project: $project_id})
        SET d:Detail
        SET d.title = $title
        SET d.text = $description
        SET d.embedding = $vector
        
        MERGE (d)-[:FOUND_ON]->(p)
        
        // Create Component Nodes (The Ingredients)
        FOREACH (mat IN $materials |
            MERGE (c:Component {name: mat.item_name, project: $project_id})
            MERGE (d)-[:CONTAINS {qty_rule: mat.qty_rule, notes: mat.notes}]->(c)
        )
        """
        
        # 5. EXECUTE THE QUERY (This is the part you asked for)
        with self.driver.session() as session:
            session.run(
                query, 
                project_id=project_id, 
                detail_key=detail_key, 
                title=title, 
                materials=clean_materials, 
                page_num=page_num,
                description=description,
                vector=vector
            )
            print(f"Graph: Added Detail BOM '{detail_key}' with Vector.")

    # --- RETRIEVAL (GraphRAG Search) ---

    def semantic_search(self, query_text, project_id, limit=3):
        """
        Finds the most relevant Definition (Rule/Detail) for a given query.
        """
        vector = self.embedder.embed_query(query_text)
        
        # query = """
        # CALL db.index.vector.queryNodes('definition_index', $limit, $vector)
        # YIELD node, score
        # WHERE node.project = $project_id
        
        # // Fetch connected components if it's a Detail
        # OPTIONAL MATCH (node)-[r:CONTAINS]->(c:Component)
        
        # RETURN 
        #     node.id as ID,
        #     node.specs as Specs,
        #     node.title as Title,
        #     collect({material: c.name, rule: r.qty_rule}) as BOM,
        #     score
        # """

        query = """
        // 1. Find the Detail Node
        CALL db.index.vector.queryNodes('definition_index', $limit, $vector)
        YIELD node, score
        WHERE node.project = $project_id
        
        // 2. Get its Components
        OPTIONAL MATCH (node)-[r:CONTAINS]->(c:Component)
        
        // 3. RECURSIVE LOOKUP: Does this component match a Schedule?
        // We look for a Schedule Rule that has a similar name to the Component
        OPTIONAL MATCH (rule:Definition:Schedule)
        WHERE rule.project = $project_id 
          AND toLower(rule.name) CONTAINS toLower(c.name) // Simple string match for now
        
        RETURN 
            node.id as ID,
            node.title as Title,
            collect({
                material: c.name, 
                qty_rule: r.qty_rule,
                linked_schedule: rule.specs // <--- THIS IS THE MISSING LINK
            }) as BOM,
            score
        """
        with self.driver.session() as session:
            result = session.run(query, vector=vector, project_id=project_id, limit=limit)
            return [record.data() for record in result]

graph_db = ConstructionGraph()
import json
import os
import time
import certifi
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from src.logger import setup_logger

load_dotenv()
logger = setup_logger(__name__)


class ConstructionGraph:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        logger.info(f"Initializing Neo4j connection | uri={uri} | user={user}")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

        # Initialize Embedding Model
        self.embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

        # Create Vector Index (Run this once)
        self.create_vector_index()

    def close(self):
        logger.info("Closing Neo4j driver")
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
    def add_text_rule(
    self,
    project_id,
    section_name,
    rule_number,
    text
):
        try:
            # 1. Create unique ID
            rule_id = f"{section_name}_{rule_number}"

            # 2. Create embedding text
            description = f"{section_name} Rule {rule_number}: {text}"

            # 3. Generate embedding
            vector = self.embedder.embed_query(description)

            query = """
            MERGE (proj:Project {id: $project_id})

            MERGE (s:Section {name: $section_name, project: $project_id})
            MERGE (proj)-[:HAS_SECTION]->(s)

            MERGE (r:Definition {id: $rule_id, project: $project_id})
            SET r:Rule
            SET r.text = $text
            SET r.order = $rule_number
            SET r.embedding = $vector

            MERGE (s)-[:HAS_RULE]->(r)
            """

            with self.driver.session() as session:
                session.run(
                    query,
                    project_id=project_id,
                    section_name=section_name,
                    rule_id=rule_id,
                    text=text,
                    rule_number=rule_number,
                    vector=vector
                )

        except Exception as e:
            logger.error(f"Failed to add text rule: {e}")
            raise


    def add_schedule_rule(
            self,
            project_id,
            schedule_name,
            symbol,
            row_data,
            columns,
            page_num,
            sheet_number
    ):
        logger.info(
    f"Adding schedule rule | project_id={project_id} | symbol={symbol} | schedule={schedule_name}"
)
        """
        Stores a rule from a schedule with Vector Embeddings.
        """
        # 1. Create Description for Embedding
        # Example: "Symbol: <1>. Schedule: Shear Wall. Specs: 5/8 bolt @ 16oc"
        try:
            # 1. Create Description
            row_json = json.dumps(row_data)
            description = f"Symbol: {symbol}. Schedule: {schedule_name}. Row: {row_json}"
            # 2. Generate Vector
            logger.debug("Generating embedding vector")
            vector = self.embedder.embed_query(description)
            logger.debug(f"Vector generated | dim={len(vector)}")

            # 3. Cypher Query
            query = """
            MERGE (proj:Project {id: $project_id})
            MERGE (p:Sheet {sheet_number: $sheet_number, project: $project_id})
            ON CREATE SET p.page_index = $page_num

            MERGE (p)-[:BELONGS_TO]->(proj)

            MERGE (d:Definition {id: $symbol, schedule: $schedule_name, project: $project_id})
            SET d:Schedule
            SET d.name = $schedule_name
            SET d.columns = $columns
            SET d.row = $row_json

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
                    row_json=row_json,
                    columns=columns,
                    page_num=page_num,
                    sheet_number=sheet_number,
                    description=description,
                    vector=vector
                )
            logger.info(
        f"Schedule rule added successfully | project_id={project_id} | symbol={symbol}"
    )
        except Exception as e:
           logger.error( f"Failed to add schedule rule | project_id={project_id} | symbol={symbol} | error={str(e)}")
           raise

    def add_detail_bom(
            self,
            project_id,
            detail_key,
            title,
            materials_list,
            page_num,
            sheet_number
    ):

        """
        Stores a Detail and its BOM, including Vector Embeddings for GraphRAG.
        """
        # 1. Create rich description for embedding
        # Example: "Detail: 7/S-3.2. Title: Ladder. Contains: MC6x15.1, L4x4"
        mat_text = ", ".join([m["item_name"] for m in materials_list])
        description = f"Detail: {detail_key}. Title: {title}. Contains: {mat_text}"

        # 2. Generate Vector
        try:
            vector = self.embedder.embed_query(description)
        except Exception as e:
            logger.error(
                f"Embedding failed | project_id={project_id} | detail_key={detail_key} | error={str(e)}"
            )
            raise

        # 3. Prepare Data for Cypher (Convert Pydantic/Dict to clean list)
        clean_materials = [
            {
                "item_name": m.get("item_name", ""),
                "qty_rule": m.get("qty_rule") or "",
                "notes": m.get("notes") or ""
            }
            for m in materials_list
        ]

        # 4. The Cypher Query
        query = """
        MERGE (proj:Project {id: $project_id})
        MERGE (p:Sheet {sheet_number: $sheet_number, project: $project_id})

        MERGE (p)-[:BELONGS_TO]->(proj)
        
        // Create/Update the Definition Node
        MERGE (d:Definition {id: $detail_key, project: $project_id})
        SET p.page_index = $page_num
        SET d:Detail
        SET d.title = $title
        SET d.text = $description
        SET d.embedding = $vector
        SET d.sheet = $sheet_number
        SET d.page = $page_num
        SET d.type = "detail"
        
        MERGE (d)-[:FOUND_ON]->(p)
        
        // Create Component Nodes (The Ingredients)
        FOREACH (mat IN $materials |
            MERGE (c:Component {name: mat.item_name, project: $project_id})
            MERGE (d)-[:CONTAINS {qty_rule: mat.qty_rule, notes: mat.notes}]->(c)
        )
        """
        for attempt in range(5):
            try:
                with self.driver.session() as session:
                    session.run(
                        query,
                        project_id=project_id,
                        detail_key=detail_key,
                        title=title,
                        materials=clean_materials,
                        page_num=page_num,
                        sheet_number=sheet_number,
                        description=description,
                        vector=vector
                    )
                logger.info(
            f"Detail BOM added | project_id={project_id} | detail_key={detail_key}"
        )
                return
            except (ServiceUnavailable, SessionExpired) as e:
                logger.warning(
    f"Neo4j retry | attempt={attempt+1}/5 | project_id={project_id} | detail_key={detail_key} | error={str(e)}"
)
                time.sleep(1)
        logger.error(
    f"Neo4j write failed after retries | project_id={project_id} | detail_key={detail_key}"
)
        raise Exception("Neo4j write failed after retries")

    # --- RETRIEVAL (GraphRAG Search) ---

    def semantic_search(self, query_text, project_id, sheet_number=None, limit=3):
        """
        Finds the most relevant Definition (Rule/Detail) for a given query.
        """
        try:
            vector = self.embedder.embed_query(query_text)
        except Exception as e:
            logger.error(
                f"Embedding failed | project_id={project_id} | query={query_text} | error={str(e)}"
            )
            raise
        logger.info(
    f"Semantic search | project_id={project_id} | query={query_text} | limit={limit}"
)

        query = """
        // 1. Find the Detail Node
        CALL db.index.vector.queryNodes('definition_index', $limit, $vector)
        YIELD node, score
        MATCH (node)-[:FOUND_ON]->(p:Sheet)
        WHERE node.project = $project_id
        AND ($sheet_number IS NULL OR p.sheet_number = $sheet_number)
        
        // 2. Get its Components
        OPTIONAL MATCH (node)-[r:CONTAINS]->(c:Component)
        
        // 3. RECURSIVE LOOKUP: Does this component match a Schedule?
        // We look for a Schedule Rule that has a similar name to the Component
        OPTIONAL MATCH (rule:Definition:Schedule)-[:FOUND_ON]->(p)
        WHERE rule.project = $project_id
        AND toLower(rule.name) CONTAINS toLower(c.name)

        RETURN 
            node.id as ID,
            node.title as Title,
            node.row as Rows,
            node.columns as Columns,
            collect({
                material: c.name,
                qty_rule: r.qty_rule,
                linked_schedule: rule.row
            }) as BOM,
            score
        """
        with self.driver.session() as session:
            result = session.run(
                query,
                vector=vector,
                project_id=project_id,
                sheet_number=sheet_number,
                limit=limit
            )
            records = list(result)
            logger.debug(f"Search results count | count={len(records)}")
            return [r.data() for r in records]


graph_db = ConstructionGraph()

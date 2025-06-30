import streamlit as st
import pandas as pd
import sqlite3
import os
import json
import re
from datetime import datetime
import google.generativeai as genai
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="🏏 IPL AI Query Agent",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .stats-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
    }
    
    .query-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .result-box {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    
    .error-box {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        color: white;
    }
    
    .sidebar .stSelectbox > label {
        font-weight: bold;
        color: #667eea;
    }
    
    .stTextInput > label {
        font-weight: bold;
        color: #667eea;
    }
    
    .example-queries {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class IPLDataManager:
    def __init__(self, data_folder: str = "data"):
        self.data_folder = data_folder
        self.db_path = "ipl_database.db"
        self.table_schemas = {}
        self.sample_data = {}
        
    def load_csv_to_sqlite(self):
        """Load all CSV files into SQLite database"""
        try:
            # Remove existing database
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            
            conn = sqlite3.connect(self.db_path)
            
            csv_files = {
                "Ball_by_Ball": "Ball_by_Ball.csv",
                "Match": "Match.csv", 
                "Player_Match": "Player_Match.csv",
                "Player": "Player.csv",
                "Season": "Season.csv",
                "Team": "Team.csv"
            }
            
            for table_name, csv_file in csv_files.items():
                file_path = os.path.join(self.data_folder, csv_file)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                    
                    # Store schema and sample data
                    self.table_schemas[table_name] = list(df.columns)
                    self.sample_data[table_name] = df.head(3).to_dict('records')
                    
                    st.success(f"✅ Loaded {table_name} with {len(df)} records")
                else:
                    st.warning(f"⚠️ File not found: {file_path}")
            
            conn.close()
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return False
    
    def get_database_schema(self) -> str:
        """Generate database schema description for AI"""
        schema_description = "IPL Database Schema:\n\n"
        
        table_descriptions = {
            "Ball_by_Ball": "Contains ball-by-ball data of matches including runs scored, extras, dismissals",
            "Match": "Contains match details including teams, venue, toss, winner, etc.",
            "Player_Match": "Contains player participation details for each match",
            "Player": "Contains player information including name, country, batting/bowling style",
            "Season": "Contains season information with awards and year",
            "Team": "Contains team information with names and short codes"
        }
        
        for table_name, columns in self.table_schemas.items():
            description = table_descriptions.get(table_name, "")
            schema_description += f"Table: {table_name}\n"
            schema_description += f"Description: {description}\n"
            schema_description += f"Columns: {', '.join(columns)}\n"
            
            if self.sample_data.get(table_name):
                schema_description += "Sample Data:\n"
                for i, row in enumerate(self.sample_data[table_name][:2]):
                    schema_description += f"  Row {i+1}: {row}\n"
            schema_description += "\n"
        
        return schema_description

class GeminiQueryAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
    def generate_sql_query(self, user_query: str, schema: str) -> Tuple[str, str]:
        """Generate SQL query from natural language using Gemini"""
        
        prompt = f"""
You are an expert SQL query generator for IPL (Indian Premier League) cricket database.

Database Schema:
{schema}

User Query: "{user_query}"

CRITICAL - Column Names and Relationships:
- Ball_by_Ball table contains: Match_Id, Innings_Id, Over_Id, Ball_Id, Team_Batting_Id, Team_Bowling_Id, Striker_Id, Striker_Batting_Position, Non_Striker_Id, Bowler_Id, Batsman_Scored, Extra_Type, Extra_Runs, Player_dissimal_Id, Dissimal_Type, Fielder_Id
- Match table contains: Match_Id, Match_Date, Team_Name_Id, Opponent_Team_Id, Season_Id, Venue_Name, Toss_Winner_Id, Toss_Decision, IS_Superover, IS_Result, Is_DuckWorthLewis, Win_Type, Won_By, Match_Winner_Id, Man_Of_The_Match_Id, First_Umpire_Id, Second_Umpire_Id, City_Name, Host_Country
- Player table contains: Player_Id, Player_Name, DOB, Batting_Hand, Bowling_Skill, Country, Is_Umpire
- Player_Match table contains: Match_Id, Player_Id, Team_Id, Is_Keeper, Is_Captain
- Season table contains: Season_Id, Season_Year, Orange_Cap_Id, Purple_Cap_Id, Man_of_the_Series_Id
- Team table contains: Team_Id, Team_Name, Team_Short_Code

IMPORTANT - For scoring/runs queries:
- Use Ball_by_Ball.Batsman_Scored for individual ball scores
- Use Ball_by_Ball.Extra_Runs for extra runs
- Total runs = Batsman_Scored + Extra_Runs
- NO column called "match_result" exists - this is WRONG!

IMPORTANT - For captain queries:
- Use Player_Match.Is_Captain = 1 to identify captains
- Join Player_Match -> Player -> Ball_by_Ball for captain batting stats
- Join Player_Match -> Match -> Season for season information

Common Query Patterns:
- For captain batting scores: 
  SELECT P.Player_Name, S.Season_Year, SUM(B.Batsman_Scored) as Total_Runs
  FROM Player P 
  JOIN Player_Match PM ON P.Player_Id = PM.Player_Id 
  JOIN Ball_by_Ball B ON PM.Player_Id = B.Striker_Id AND PM.Match_Id = B.Match_Id
  JOIN Match M ON PM.Match_Id = M.Match_Id 
  JOIN Season S ON M.Season_Id = S.Season_Id 
  WHERE PM.Is_Captain = 1

- For team performance: Use Match.Match_Winner_Id, Match.Won_By
- For player stats: Use Ball_by_Ball.Batsman_Scored for runs, count dismissals from Ball_by_Ball
- For match details: Use proper column names from Match table

Instructions:
1. Generate ONLY a valid SQLite SQL query that answers the user's question
2. Use EXACT column names as specified above - do not invent column names
3. Use proper table joins when needed
4. Use LIMIT clause for large result sets (default LIMIT 50 unless user asks for specific number)
5. Handle case-insensitive searches using LOWER() function
6. Use appropriate aggregation functions (COUNT, SUM, AVG, MAX, MIN) when needed
7. For player names, team names, use LIKE operator with % wildcards for partial matches
8. Return only the SQL query, no explanation or markdown formatting

Generate SQL Query:
"""
        
        try:
            response = self.model.generate_content(prompt)
            sql_query = response.text.strip()
            
            # Clean the SQL query
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            sql_query = sql_query.strip()
            
            return sql_query, "Success"
            
        except Exception as e:
            return "", f"Error generating SQL: {str(e)}"
    
    def validate_sql_query(self, sql_query: str, schema: str) -> Tuple[bool, str]:
        """Validate SQL query against schema"""
        try:
            # Define valid column names for each table
            valid_columns = {
                'Ball_by_Ball': ['Match_Id', 'Innings_Id', 'Over_Id', 'Ball_Id', 'Team_Batting_Id', 
                               'Team_Bowling_Id', 'Striker_Id', 'Striker_Batting_Position', 
                               'Non_Striker_Id', 'Bowler_Id', 'Batsman_Scored', 'Extra_Type', 
                               'Extra_Runs', 'Player_dissimal_Id', 'Dissimal_Type', 'Fielder_Id'],
                'Match': ['Match_Id', 'Match_Date', 'Team_Name_Id', 'Opponent_Team_Id', 'Season_Id', 
                         'Venue_Name', 'Toss_Winner_Id', 'Toss_Decision', 'IS_Superover', 'IS_Result', 
                         'Is_DuckWorthLewis', 'Win_Type', 'Won_By', 'Match_Winner_Id', 'Man_Of_The_Match_Id', 
                         'First_Umpire_Id', 'Second_Umpire_Id', 'City_Name', 'Host_Country'],
                'Player': ['Player_Id', 'Player_Name', 'DOB', 'Batting_Hand', 'Bowling_Skill', 'Country', 'Is_Umpire'],
                'Player_Match': ['Match_Id', 'Player_Id', 'Team_Id', 'Is_Keeper', 'Is_Captain'],
                'Season': ['Season_Id', 'Season_Year', 'Orange_Cap_Id', 'Purple_Cap_Id', 'Man_of_the_Series_Id'],
                'Team': ['Team_Id', 'Team_Name', 'Team_Short_Code']
            }
            
            # Check for common invalid column names
            invalid_patterns = [
                'match_result', 'total_score', 'score', 'runs_scored'
            ]
            
            sql_lower = sql_query.lower()
            for pattern in invalid_patterns:
                if pattern in sql_lower and 'batsman_scored' not in sql_lower:
                    return False, f"Invalid column reference: {pattern}. Use 'Batsman_Scored' for runs."
            
            return True, "Valid"
            
        except Exception as e:
            return True, "Validation skipped"  # Continue if validation fails
    
    def regenerate_sql_with_correction(self, user_query: str, schema: str, error_message: str) -> Tuple[str, str]:
        """Regenerate SQL query with error correction"""
        
        correction_prompt = f"""
The previous SQL query failed with error: {error_message}

Database Schema:
{schema}

User Query: "{user_query}"

CRITICAL CORRECTIONS NEEDED:
1. There is NO column called "match_result" - this does not exist!
2. For batting scores, use Ball_by_Ball.Batsman_Scored 
3. For total runs per player: SUM(Ball_by_Ball.Batsman_Scored)
4. For captains: use Player_Match.Is_Captain = 1
5. Use exact column names from schema only

For the user query about captain scores by season, you need:
- Player names from Player table
- Captain identification from Player_Match.Is_Captain = 1  
- Batting scores from Ball_by_Ball.Batsman_Scored
- Season information from Season.Season_Year

Generate a corrected SQL query using ONLY valid column names:
"""
        
        try:
            response = self.model.generate_content(correction_prompt)
            sql_query = response.text.strip()
            
            # Clean the SQL query
            sql_query = re.sub(r'```sql\n?', '', sql_query)
            sql_query = re.sub(r'```\n?', '', sql_query)
            sql_query = sql_query.strip()
            
            return sql_query, "Corrected"
            
        except Exception as e:
            return "", f"Error in correction: {str(e)}"
    
    def execute_sql_query(self, sql_query: str, db_path: str) -> Tuple[pd.DataFrame, str]:
        """Execute SQL query and return results"""
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            
            return df, "Success"
            
        except Exception as e:
            return pd.DataFrame(), f"SQL Execution Error: {str(e)}"
    
    def generate_natural_response(self, user_query: str, sql_query: str, results_df: pd.DataFrame) -> str:
        """Generate natural language response from query results"""
        
        if results_df.empty:
            return "No data found matching your query."
        
        # Convert DataFrame to string representation (limited rows for context)
        results_summary = results_df.head(10).to_string(index=False)
        total_rows = len(results_df)
        
        prompt = f"""
You are an IPL cricket expert providing insights based on data analysis.

User asked: "{user_query}"
SQL Query used: {sql_query}
Query returned {total_rows} rows.

Sample Results:
{results_summary}

Instructions:
1. Provide a natural, conversational response about the results
2. Highlight key insights and interesting findings
3. Use cricket terminology appropriately
4. If there are many results, summarize the key points
5. Be specific with numbers and statistics
6. Keep the response informative but concise

Generate Natural Response:
"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
            
        except Exception as e:
            return f"Generated response based on {total_rows} records from the database."

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏏 IPL AI Query Agent</h1>
        <p>Ask questions about IPL data in natural language and get intelligent responses!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        st.error("❌ GEMINI_API_KEY not found in .env file. Please add your API key to the .env file.")
        st.info("Create a .env file in your project directory with: GEMINI_API_KEY=your_api_key_here")
        return
    
    # Sidebar for configuration
    st.sidebar.markdown("## ⚙️ Configuration")
    st.sidebar.success("✅ API Key loaded from .env file")
    
    # Data folder input
    data_folder = st.sidebar.text_input(
        "📁 Data Folder Path:",
        value="data",
        help="Path to folder containing CSV files"
    )
    
    # Initialize components
    data_manager = IPLDataManager(data_folder)
    query_agent = GeminiQueryAgent(api_key)
    
    # Load data button
    if st.sidebar.button("🔄 Load Data", type="primary"):
        with st.spinner("Loading CSV files into database..."):
            if data_manager.load_csv_to_sqlite():
                st.sidebar.success("✅ Data loaded successfully!")
                st.session_state.data_loaded = True
            else:
                st.sidebar.error("❌ Failed to load data")
                return
    
    # Check if data is loaded
    if not hasattr(st.session_state, 'data_loaded') or not st.session_state.data_loaded:
        st.info("👆 Please load the data first using the sidebar.")
        return
    
    # Example queries
    st.markdown("""
    <div class="example-queries">
        <h3>💡 Example Queries You Can Ask:</h3>
        <ul>
            <li>Show me the name of all captains with score by each season</li>
            <li>Who won the most matches in IPL 2008?</li>
            <li>Show me top run scorers in all IPL seasons</li>
            <li>Which team has the best win percentage?</li>
            <li>Who took the most wickets in IPL history?</li>
            <li>Show matches played at M Chinnaswamy Stadium</li>
            <li>Which player has played for the most teams?</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Query input
    st.markdown('<div class="query-box">', unsafe_allow_html=True)
    user_query = st.text_input(
        "🤔 Ask your IPL question:",
        placeholder="e.g., Show me the name of all captains with score by each season",
        key="user_query"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process query
    if st.button("🚀 Get Answer", type="primary", use_container_width=True):
        if not user_query.strip():
            st.warning("⚠️ Please enter a question!")
            return
        
        with st.spinner("🤖 AI is processing your query..."):
            # Generate schema
            schema = data_manager.get_database_schema()
            
            # Generate SQL query
            sql_query, sql_status = query_agent.generate_sql_query(user_query, schema)
            
            if sql_status != "Success":
                st.error(f"❌ {sql_status}")
                return
            
            # Validate SQL query
            is_valid, validation_message = query_agent.validate_sql_query(sql_query, schema)
            
            if not is_valid:
                st.warning(f"⚠️ Query validation failed: {validation_message}")
                st.info("🔄 Attempting to correct the query...")
                
                # Try to regenerate with correction
                sql_query, correction_status = query_agent.regenerate_sql_with_correction(
                    user_query, schema, validation_message
                )
                
                if correction_status != "Corrected":
                    st.error(f"❌ Failed to correct query: {correction_status}")
                    return
            
            # Show generated SQL
            with st.expander("🔍 Generated SQL Query"):
                st.code(sql_query, language="sql")
            
            # Execute query
            results_df, exec_status = query_agent.execute_sql_query(sql_query, data_manager.db_path)
            
            if exec_status != "Success":
                st.error(f"❌ {exec_status}")
                
                # Try one more correction attempt
                st.info("🔄 Attempting to fix the SQL query...")
                corrected_sql, correction_status = query_agent.regenerate_sql_with_correction(
                    user_query, schema, exec_status
                )
                
                if correction_status == "Corrected":
                    st.info("✅ Query corrected, trying again...")
                    with st.expander("🔍 Corrected SQL Query"):
                        st.code(corrected_sql, language="sql")
                    
                    results_df, exec_status = query_agent.execute_sql_query(corrected_sql, data_manager.db_path)
                    
                    if exec_status != "Success":
                        st.error(f"❌ Still failed after correction: {exec_status}")
                        return
                else:
                    st.error("❌ Unable to correct the query automatically.")
                    return
            
            # Display results
            if not results_df.empty:
                # Generate natural response
                natural_response = query_agent.generate_natural_response(user_query, sql_query, results_df)
                
                # Show AI response
                st.markdown(f"""
                <div class="result-box">
                    <h3>🤖 AI Response:</h3>
                    <p>{natural_response}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show data table
                st.markdown("### 📊 Query Results:")
                st.dataframe(results_df, use_container_width=True)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"""
                    <div class="stats-card">
                        <h4>Rows Returned</h4>
                        <h2>{len(results_df)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="stats-card">
                        <h4>Columns</h4>
                        <h2>{len(results_df.columns)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="stats-card">
                        <h4>Query Time</h4>
                        <h2>< 1s</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"ipl_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("ℹ️ No results found for your query.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏏 IPL AI Query Agent | Powered by Gemini AI | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
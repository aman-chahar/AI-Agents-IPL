# AI-Agents-IPL
AI Agents to chat with IPL Dataset
Web app Link: https://ai-agents-ipl-kbppwdvhcncsu3p5q8cebn.streamlit.app/


# 🏏 IPL AI Query Agent

A Streamlit app that lets you ask questions about IPL cricket data in plain English. Gemini converts your question into SQL, SQLite runs it, and Gemini summarizes the results.

## Features

- Natural language to SQL using Google Gemini
- Loads IPL CSV files into a local SQLite database
- Automatic SQL validation and one retry on errors
- AI-written summary, results table, and CSV download


## Usage

1. Click **🔄 Load Data** in the sidebar (builds `ipl_database.db`).
2. Type a question, for example:
   - *Show me the name of all captains with score by each season*
   - *Which team has the best win percentage?*
   - *Show matches played at M Chinnaswamy Stadium*
3. Click **🚀 Get Answer** to see the generated SQL, the AI summary, and the results table.

## How it works

Question → Schema + prompt → Gemini (SQL) → Validate → SQLite → Gemini (summary) → UI

## Run
streamlit run app.py


## Example Queries You Can Ask:
1. Show me the name of all captains with score by each season
2. Who won the most matches in IPL 2008?
3. Show me top run scorers in all IPL seasons
4. Which team has the best win percentage?
5. Who took the most wickets in IPL history?
6. Show matches played at M Chinnaswamy Stadium
7. Which player has played for the most teams?

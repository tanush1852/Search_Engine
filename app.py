import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import time
import json
from typing import Dict, Any

# Import your research system
from research_system import run_research_system

def main():
    st.set_page_config(page_title="Deep Research AI System", layout="wide")

    st.title("🔍 Deep Research AI System")
    st.markdown("An agentic system for comprehensive web research with fact verification")

    # Query input
    with st.form("research_form"):
        query = st.text_area("Research Question:", height=100)
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.form_submit_button("Start Research")
        with col2:
            st.caption("This will initiate the research process using multiple AI agents.")

    if submit_button and query:
        # Run the research process
        with st.spinner("Research in progress - this may take a few minutes..."):
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Mock research stages for visual feedback
            stages = [
                "Planning research questions...",
                "Gathering information from web sources...",
                "Evaluating source credibility...",
                "Verifying facts across multiple sources...",
                "Drafting comprehensive answer...",
                "Critiquing and improving answer...",
                "Creating summaries...",
                "Finalizing results..."
            ]
            
            # Show progress through stages
            for i, stage in enumerate(stages):
                status_text.text(stage)
                progress_bar.progress((i + 1) / len(stages))
                time.sleep(0.5)  # Simulated delay
            
            # Actual research happens here
            result = run_research_system(query)
            
            # Mark completion
            progress_bar.progress(100)
            status_text.text("Research complete!")
        
        # Display results in tabs
        st.subheader("Research Results")
        tabs = st.tabs(["Summary", "Detailed Answer", "Source Analysis", "Fact Verification", "Research Process"])
        
        # Summary Tab
        with tabs[0]:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Executive Summary")
                st.write(result["summaries"]["executive"])
                
                st.subheader("Brief Summary")
                st.write(result["summaries"]["brief"])
                
                st.subheader("One-Sentence Summary")
                st.write(result["summaries"]["one_sentence"])
            
            with col2:
                # Key metrics
                st.subheader("Research Metrics")
                
                # Source credibility chart
                source_data = result["visualization_data"]["sources"]["credibility"]
                fig = px.pie(
                    names=["High Quality", "Medium Quality", "Low Quality"],
                    values=[source_data["high"], source_data["medium"], source_data["low"]],
                    title="Source Quality Distribution",
                    color_discrete_sequence=px.colors.sequential.Viridis,
                    hole=0.4
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Fact verification chart
                verification = result["visualization_data"]["facts"]["verification"]
                fig = px.bar(
                    x=["Confirmed", "Inconclusive", "Contradicted"],
                    y=[verification["confirmed"], verification["inconclusive"], verification["contradicted"]],
                    title="Fact Verification Results",
                    color_discrete_sequence=["green", "orange", "red"]
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Answer Tab
        with tabs[1]:
            st.markdown(result["final_answer"])
        
        # Source Analysis Tab
        with tabs[2]:
            st.subheader("Source Credibility Analysis")
            
            sources = result["visualization_data"]["sources"]["details"]
            if sources:
                # Create a DataFrame for the source assessments
                source_df = pd.DataFrame([
                    {
                        "Source": s.get("url", "Unknown"),
                        "Overall Credibility": s.get("overall_credibility", 5),
                        "Domain Authority": s.get("domain_authority_score", "N/A"),
                        "Recency": s.get("recency_score", "N/A"),
                        "Expertise": s.get("expertise_score", "N/A"),
                        "Content Quality": s.get("content_quality_score", "N/A"),
                        "Bias Level": s.get("bias_assessment", "N/A")
                    }
                    for s in sources
                ])
                
                # Calculate average scores
                avg_scores = {
                    "Overall": source_df["Overall Credibility"].mean(),
                    "Domain Authority": pd.to_numeric(source_df["Domain Authority"], errors="coerce").mean(),
                    "Recency": pd.to_numeric(source_df["Recency"], errors="coerce").mean(),
                    "Content Quality": pd.to_numeric(source_df["Content Quality"], errors="coerce").mean()
                }
                
                # Display radar chart of average scores
                categories = list(avg_scores.keys())
                values = list(avg_scores.values())
                
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Average Source Quality'
                ))
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 10]
                        )),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display source table
                st.dataframe(source_df)
            else:
                st.info("No source assessment data available.")
        
        # Fact Verification Tab
        with tabs[3]:
            st.subheader("Fact Verification Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Verified Facts")
                verified_facts = result["visualization_data"]["facts"]["verified_facts"]
                
                for i, fact in enumerate(verified_facts):
                    with st.expander(f"Fact {i+1} - {fact.get('verification_status', 'Unknown')}"):
                        st.write(f"**Claim:** {fact.get('fact', '')}")
                        st.write(f"**Confidence:** {fact.get('confidence', 'N/A')}")
                        st.write(f"**Source Credibility:** {fact.get('source_credibility_avg', 'N/A')}")
                        
                        if fact.get("supporting_sources"):
                            st.write("**Supporting Sources:**")
                            for source in fact["supporting_sources"]:
                                st.write(f"- {source}")
                        
                        if fact.get("notes"):
                            st.write(f"**Notes:** {fact['notes']}")
            
            with col2:
                st.subheader("Verification Issues")
                issues = result["visualization_data"]["facts"]["verification_issues"]
                
                if issues:
                    for i, issue in enumerate(issues):
                        with st.expander(f"Issue {i+1}"):
                            st.write(f"**Claim:** {issue.get('fact', '')}")
                            st.write(f"**Status:** {issue.get('verification_status', 'Unknown')}")
                            st.write(f"**Confidence:** {issue.get('confidence', 'N/A')}")
                            
                            if issue.get("contradicting_sources"):
                                st.write("**Contradicting Sources:**")
                                for source in issue["contradicting_sources"]:
                                    st.write(f"- {source}")
                            
                            if issue.get("notes"):
                                st.write(f"**Notes:** {issue['notes']}")
                else:
                    st.success("No verification issues found!")
        
        # Research Process Tab
        with tabs[4]:
            st.subheader("Research Process")
            
            st.markdown("### Research Questions")
            for i, question in enumerate(result["visualization_data"].get("research_plan", [])):
                st.write(f"{i+1}. {question}")
            
            # Timeline visualization
            st.markdown("### Research Timeline")
            
            # Create timeline data
            timeline_data = {
                "Stage": [
                    "Research Planning",
                    "Information Gathering",
                    "Source Assessment",
                    "Fact Verification",
                    "Content Drafting",
                    "Critical Review",
                    "Content Revision",
                    "Summarization"
                ],
                "Start": list(range(8)),
                "End": list(range(1, 9))
            }
            
            fig = px.timeline(
                timeline_data, 
                x_start="Start", 
                x_end="End", 
                y="Stage",
                color="Stage",
                title="Research Process Timeline"
            )
            fig.update_layout(xaxis_title="Time (relative)")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
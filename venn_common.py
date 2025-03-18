import streamlit as st
import fitz  
from io import BytesIO
import pandas as pd
from itertools import combinations
import matplotlib.pyplot as plt
from venny4py.venny4py import venny4py  # Import venny4py for 4-set Venn diagrams
import matplotlib_venn as venn  # Correct import for venn library
from fuzzywuzzy import fuzz  # Import fuzzywuzzy for fuzzy string matching
import time  # Import time module to track execution time
import datetime
import re  # For extracting years
from upsetplot import UpSet, from_memberships

# Disable the deprecation warning globally
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set the threshold at the top of the code (you can change this value easily here)
THRESHOLD = 90 # Set your default threshold here

# Function to normalize publication titles
def clean_title(title):
    if not title or not isinstance(title, str):
        return ""
    # Remove punctuation, normalize spaces, and convert to lowercase
    title = re.sub(r'[^\w\s]', '', title)  # Remove punctuation
    return " ".join(title.lower().strip().split())

# Strip the .pdf extension from file name
def clean_filename(file_name):
    return file_name.rsplit('.', 1)[0]  # Remove the .pdf extension

def is_google_scholar(file_object):
    if file_object is None or file_object.getbuffer().nbytes == 0:
        file_name = getattr(file_object, "name", "Unknown file")
        st.error(f"The uploaded file '{file_name}' is empty or invalid.")
        return False

    file_object.seek(0)
    pdf_document = fitz.open(stream=file_object.read(), filetype="pdf")
    page1 = pdf_document.load_page(0)

    topmost_y = float('inf')
    topmost_text = ""

    for block in page1.get_text("dict")["blocks"]:
        block_type = block.get("type", 0)
        if block_type == 0: 
            block_bbox = block.get("bbox", [])
            if block_bbox and block_bbox[1] < topmost_y:
                topmost_y = block_bbox[1]
                topmost_text = " ".join(
                    span['text'] for line in block.get("lines", [])
                    for span in line.get("spans", [])
                )

    pdf_document.close()
    return "Google Scholar" in topmost_text

def is_blue(color_int):
    r = (color_int >> 16) & 0xFF
    g = (color_int >> 8) & 0xFF
    b = color_int & 0xFF
    return b > r and b > g

# def extract_titles_citations_years(file_object):
#     titles = []
#     citations = []
#     years = []
#     current_title = ""
#     found_first_title = False  

#     if file_object is None or file_object.getbuffer().nbytes == 0:
#         return []

#     file_object.seek(0)
#     pdf_document = fitz.open(stream=file_object.read(), filetype="pdf")

#     try:
#         for page_number in range(len(pdf_document)):
#             page = pdf_document.load_page(page_number)
#             text_blocks = page.get_text("dict")["blocks"]

#             # First pass: Extract titles (in blue)
#             for block in text_blocks:
#                 if "lines" in block:
#                     for line in block["lines"]:
#                         for span in line["spans"]:
#                             if is_blue(span['color']):  
#                                 current_title += span['text'] + " "  
#                                 found_first_title = True  
#                             else:
#                                 if current_title.strip():  
#                                     titles.append(clean_title(current_title.strip()))
#                                     current_title = ""  

#             # Second pass: Extract years and citations using block structure
#             text_lines = page.get_text("text").split("\n")
#             for i in range(len(text_lines) - 1):
#                 line = text_lines[i].strip()
#                 next_line = text_lines[i + 1].strip()
#                 if found_first_title and re.match(r"\b(20\d{2}|19\d{2})\b", next_line):
#                     year_match = re.search(r"\b(20\d{2}|19\d{2})\b", next_line)
#                     year = year_match.group(0) if year_match else "N/A"
#                     years.append(year)

#                     # Find the citation by looking at the block containing the year
#                     citation_found = False
#                     for block in text_blocks:
#                         if "lines" in block:
#                             block_text = " ".join(
#                                 span['text'] for line in block["lines"]
#                                 for span in line["spans"]
#                             )
#                             if year in block_text:
#                                 # Look for a number in the previous line or block
#                                 block_lines = [
#                                     span['text'].strip()
#                                     for line in block["lines"]
#                                     for span in line["spans"]
#                                 ]
#                                 for j in range(len(block_lines)):
#                                     if year in block_lines[j]:
#                                         # Check the previous line for a citation
#                                         if j > 0:
#                                             prev_line = block_lines[j - 1]
#                                             if prev_line.isdigit():
#                                                 citations.append(int(prev_line))
#                                                 citation_found = True
#                                                 break
#                                         # If not found, check the line itself for a number
#                                         line_parts = block_lines[j].split()
#                                         for part in line_parts:
#                                             if part.isdigit() and part != year:
#                                                 citations.append(int(part))
#                                                 citation_found = True
#                                                 break
#                                 break
#                     if not citation_found:
#                         citations.append(0)

#         # Ensure lists are the same length
#         max_length = max(len(titles), len(citations), len(years))
#         while len(titles) < max_length:
#             titles.append("N/A")
#         while len(citations) < max_length:
#             citations.append(0)
#         while len(years) < max_length:
#             years.append("N/A")

#     finally:
#         pdf_document.close()

#     extracted_data = list(zip(titles, years, citations))
    
#     # Debug: Print raw extracted data
#     st.write(f"Debug - Raw Extracted Data for {clean_filename(file_object.name)}:")
#     for title, year, citation in extracted_data:
#         st.write(f"  Title: {title}, Year: {year}, Citation: {citation}")

#     return extracted_data


def extract_titles_citations_years(file_object):
    """
    Extracts (Title, Year, Citations) from a PDF.
    Uses pdfplumber to extract all text (saved for debugging) and a regex that
    allows an optional citation field. Publications missing a citation are assigned 0.
    Expected line formats:
       Economic analysis of integrated ... 611 2011
       Some Publication Title Without Citation 2015
    """
    if file_object is None or file_object.getbuffer().nbytes == 0:
        return []
    file_object.seek(0)
    try:
        import pdfplumber
        with pdfplumber.open(file_object) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF with pdfplumber: {e}")
        return []
    
    # Save extracted text for debugging
    with open("extracted_text.txt", "w", encoding="utf-8") as f:
        f.write(full_text)
    
    publications = []
    # Modified regex: citation field is optional
    pattern = re.compile(r'^(.*?)\s+(?:(\d+)\s+)?(19\d{2}|20\d{2})$')
    lines = full_text.splitlines()
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # Skip header or non-publication lines
        if any(keyword in line_stripped.upper() for keyword in ["TITLE", "CITED BY", "H-INDEX", "I10-INDEX"]):
            continue
        if "articles" in line_stripped.lower():
            continue
        match = pattern.match(line_stripped)
        if match:
            raw_title = match.group(1).strip()
            raw_citations = match.group(2)  # May be None
            raw_year = match.group(3).strip()
            try:
                citations = int(raw_citations) if raw_citations is not None else 0
                year = int(raw_year)
                if year < 1900 or year > 2050:
                    continue
                if len(raw_title.split()) < 3:
                    continue
                publications.append((clean_title(raw_title), str(year), citations))
            except ValueError:
                continue

    # st.write(f"Debug - Raw Extracted Data for {clean_filename(getattr(file_object, 'name', 'Unknown'))}:")
    # for title, year, citation in publications:
    #     st.write(f"  Title: {title}, Year: {year}, Citation: {citation}")
    return publications

# Fuzzy string matching to find similar titles between two sets
def find_similar_titles(set1, set2, threshold=THRESHOLD):  
    matches = set()
    for title1 in set1:
        for title2 in set2:
            if title1 == "N/A" or title2 == "N/A":
                continue
            if fuzz.ratio(title1, title2) >= threshold:
                matches.add(title1)  # Store only one version
    return matches

# Add a button to run the simulation
st.title("Google Scholar PubMatch")

# Add the attribution text below the title
st.markdown(
    '"This common publications matching tool is implemented based on the work done by Rohit Ramachandran, Rutgers University, NJ, USA"'
)

# Add instructions banner
st.markdown("""
### Instructions:
1. Upload at least 2 Google Scholar PDFs. Ensure all the papers are printed to PDF. See 'show more' at the end of the google scholar link.
2. The algorithm will check for combinations of common publications across 2 to N researchers, where N ~=8 before memory limitations hit. N=8 has 247 combinations.
3. Once the analysis is complete, download the comparison results as a CSV file.
4. You can also view the table, Upset plot and Venn diagram showing the overlap of publications between researchers. 
5. Venn diagrams are absent for 5 or more researchers. 
""")

# Get the current year dynamically
current_year = datetime.datetime.now().year

# Dropdown for year filtering
year_options = ["All Years"] + list(range(current_year, 1950, -1))  
selected_years = st.multiselect("Select years to include:", year_options, default=["All Years"])  

# If "All Years" is selected, include all years
if "All Years" in selected_years:
    selected_years = set(map(str, range(current_year, 1950, -1)))  
else:
    selected_years = set(map(str, [year for year in selected_years if year != "All Years"]))  

uploaded_files = st.file_uploader(
    "Upload Google Scholar PDFs of researchers",
    type=["pdf"],
    accept_multiple_files=True
)

# Add a button for running the simulation
run_simulation_button = st.button("Run Simulation")

# If the button is pressed, start the simulation
if run_simulation_button:
    # Record start time
    start_time = time.time()

    if uploaded_files:
        researcher_data = {}

        for file in uploaded_files:
            if is_google_scholar(file):
                researcher_name = clean_filename(file.name)  
                extracted_data = extract_titles_citations_years(file)
                # Filter titles by the selected years
                filtered_titles = { (title, year, int(citations) if citations != "N/A" else 0) for title, year, citations in extracted_data if year in selected_years}
                researcher_data[researcher_name] = filtered_titles

        if len(researcher_data) > 1:
            st.subheader("Common Publications Between Researchers")

            all_sets = {name: set(t[0] for t in titles if t[0] != "N/A") for name, titles in researcher_data.items()}
            all_data_with_details = {name: titles for name, titles in researcher_data.items()}
            comparisons = []

            # Generate all subset combinations (from pairs to all researchers)
            for r in range(2, len(all_sets) + 1):  
                for comb in combinations(all_sets.keys(), r):
                    common_titles = all_sets[comb[0]]
                    for researcher in comb[1:]:
                        common_titles = find_similar_titles(common_titles, all_sets[researcher])  
                    
                    # Calculate total citations for common publications (for the combination table)
                    total_citations = 0
                    for title in common_titles:
                        if title:
                            # Collect citations from all researchers in the combination
                            citation_values = []
                            for researcher in comb:
                                for t, y, c in all_data_with_details[researcher]:
                                    if t == "N/A":
                                        continue
                                    if fuzz.ratio(t, title) >= THRESHOLD:
                                        c_int = int(c) if c != "N/A" else 0
                                        citation_values.append((researcher, c_int))
                                        break
                                else:
                                    citation_values.append((researcher, 0))
                            # Use the minimum citation value (if one is 0, use 0)
                            if citation_values:
                                chosen_citation = min(citation[1] for citation in citation_values)
                                total_citations += chosen_citation
                    
                    # Join common titles into a string
                    common_titles_str = "|".join(common_titles)
                    
                    comparisons.append({
                        "Combinations": " ↔ ".join(comb),  
                        "No. of common publications": len(common_titles),
                        "Total Common Citations": total_citations,
                        "Common Publications": common_titles_str,  
                    })

                    # Debug: Print common publications and their citations for this combination
                    #st.write(f"Debug - Common Publications for {comb}:")
                    for title in common_titles:
                        if title:
                            citation_values = []
                            for researcher in comb:
                                for t, y, c in all_data_with_details[researcher]:
                                    if t == "N/A":
                                        continue
                                    if fuzz.ratio(t, title) >= THRESHOLD:
                                        c_int = int(c) if c != "N/A" else 0
                                        citation_values.append((researcher, c_int))
                                        break
                                else:
                                    citation_values.append((researcher, 0))
                            chosen_citation = min(citation[1] for citation in citation_values) if citation_values else 0
                            #st.write(f"  Title: {title}, Chosen Citation: {chosen_citation}, All Citations: {citation_values}")

            # Create the DataFrame with the common publications count and citations
            df_comparisons = pd.DataFrame(comparisons)
            df_comparisons.index = range(1, len(df_comparisons) + 1)

            # Display the table without "Common Publications"
            st.dataframe(
                df_comparisons.drop(columns=["Common Publications"]).style.set_properties(**{
                    'text-align': 'left',
                    'white-space': 'pre-wrap'
                }),
                height=(50 + len(df_comparisons) * 35),  
                width=800
            )

            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            # Calculate total common publications and citations for each researcher
            researcher_totals = {name: {"common_titles_with_citations": set()} for name in researcher_data.keys()}
            
            # Collect all common titles and their citations per researcher
            for _, row in df_comparisons.iterrows():
                researchers_in_comb = row["Combinations"].split(" ↔ ")
                common_titles = row["Common Publications"].split("|")
                
                for researcher in researchers_in_comb:
                    for title in common_titles:
                        if title:
                            # Determine the consistent citation value for this title
                            citation_values = []
                            for r in researchers_in_comb:
                                for t, y, c in all_data_with_details[r]:
                                    if t == "N/A":
                                        continue
                                    if fuzz.ratio(t, title) >= THRESHOLD:
                                        c_int = int(c) if c != "N/A" else 0
                                        citation_values.append(c_int)
                                        break
                                else:
                                    citation_values.append(0)
                            chosen_citation = min(citation_values) if citation_values else 0
                            researcher_totals[researcher]["common_titles_with_citations"].add((title, chosen_citation))

            # Calculate total publications and citations based on unique common titles
            for researcher in researcher_totals:
                unique_titles = set(title for title, citation in researcher_totals[researcher]["common_titles_with_citations"])
                total_citations = 0
                for title, citation in researcher_totals[researcher]["common_titles_with_citations"]:
                    total_citations += citation
                researcher_totals[researcher]["publications"] = len(unique_titles)
                researcher_totals[researcher]["citations"] = total_citations

                # Debug: Print unique common publications and their citations for this researcher
                # st.write(f"Debug - Unique Common Publications for {researcher}:")
                # for title, citation in researcher_totals[researcher]["common_titles_with_citations"]:
                #     st.write(f"  Title: {title}, Citation: {citation}")

            # Create a DataFrame for total common publications and citations
            df_totals = pd.DataFrame(
                [(name, data["publications"], data["citations"]) for name, data in researcher_totals.items()],
                columns=["Researcher", "Total Common Publications", "Total Common Citations"]
            )

            # Reset the index to start from 1
            df_totals.index = range(1, len(df_totals) + 1)

            # Display the updated table
            st.subheader("Total Common Publications and Citations per Researcher")
            st.dataframe(
                df_totals.style.set_properties(**{
                    'text-align': 'left',
                    'white-space': 'pre-wrap'
                }),
                height=(50 + len(df_totals) * 35),
                width=800
            )

            # Provide option to download the table as CSV
            csv_data = convert_df(df_comparisons)
            st.download_button(
                label="Download Comparison as CSV",
                data=csv_data,
                file_name="publication_comparisons.csv",
                mime="text/csv"
            )
            
            # Generate the UpSet plot from the comparisons data dynamically
            st.subheader("UpSet Plot of Common Publications")
            researcher_names = list(researcher_data.keys())
            upset_data = {}

            for _, row in df_comparisons.iterrows():
                combination = row["Combinations"]
                common_publications_count = row["No. of common publications"]
                researchers_in_comb = combination.split(" ↔ ")
                membership = tuple(researcher in researchers_in_comb for researcher in researcher_names)
                if membership in upset_data:
                    upset_data[membership] += common_publications_count
                else:
                    upset_data[membership] = common_publications_count

            index = pd.MultiIndex.from_tuples(upset_data.keys(), names=researcher_names)
            data_counts = pd.Series(list(upset_data.values()), index=index)

            upset = UpSet(data_counts, subset_size="auto", show_counts="%d", sort_by="cardinality", facecolor="blue")
            fig = plt.figure(figsize=(12, 8))
            upset.plot(fig=fig)
            st.pyplot(fig)
                            
            # Only plot the Venn diagram if there are 2, 3, or 4 researchers
            if len(researcher_data) > 1:
                if len(researcher_data) > 4:
                    st.warning("Venn diagrams are only available for up to 4 researchers. Skipping Venn diagram.")
                else:
                    researcher_names = list(all_sets.keys())
                    total_publications = {name: len(all_sets[name]) for name in researcher_names}

                    intersection_counts = {}
                    for r in range(1, len(researcher_names) + 1):
                        for comb in combinations(researcher_names, r):
                            intersection_counts[comb] = 0

                    for _, row in df_comparisons.iterrows():
                        researchers_in_comb = tuple(row["Combinations"].split(" ↔ "))
                        count = row["No. of common publications"]
                        intersection_counts[researchers_in_comb] = count

                    singleton_counts = {}
                    for name in researcher_names:
                        total = total_publications[name]
                        for comb, count in intersection_counts.items():
                            if name in comb:
                                total -= count
                        singleton_counts[name] = max(total, 0)

                    if len(researcher_names) == 2:
                        name1, name2 = researcher_names
                        venn_data = [
                            singleton_counts[name1],  
                            singleton_counts[name2],  
                            intersection_counts[(name1, name2)]  
                        ]
                        fig, ax = plt.subplots(figsize=(6, 6))
                        v = venn.venn2(
                            subsets=venn_data,
                            set_labels=researcher_names
                        )
                        st.pyplot(fig)

                    elif len(researcher_names) == 3:
                        name1, name2, name3 = researcher_names
                        venn_data = [
                            singleton_counts[name1],  
                            singleton_counts[name2],  
                            intersection_counts[(name1, name2)],  
                            singleton_counts[name3],  
                            intersection_counts[(name1, name3)],  
                            intersection_counts[(name2, name3)],  
                            intersection_counts[(name1, name2, name3)]  
                        ]
                        fig, ax = plt.subplots(figsize=(6, 6))
                        v = venn.venn3(
                            subsets=venn_data,
                            set_labels=researcher_names
                        )
                        st.pyplot(fig)

                    elif len(researcher_names) == 4:
                        name1, name2, name3, name4 = researcher_names
                        venn_sets = {name: set() for name in researcher_names}
                        for name in researcher_names:
                            count = singleton_counts[name]
                            venn_sets[name].update([f"{name}_unique_{i}" for i in range(count)])
                        
                        for (n1, n2) in combinations(researcher_names, 2):
                            count = intersection_counts[(n1, n2)]
                            titles = [f"{n1}_{n2}_{i}" for i in range(count)]
                            venn_sets[n1].update(titles)
                            venn_sets[n2].update(titles)
                        
                        for (n1, n2, n3) in combinations(researcher_names, 3):
                            count = intersection_counts.get((n1, n2, n3), 0)
                            titles = [f"{n1}_{n2}_{n3}_{i}" for i in range(count)]
                            venn_sets[n1].update(titles)
                            venn_sets[n2].update(titles)
                            venn_sets[n3].update(titles)
                        
                        count = intersection_counts.get((name1, name2, name3, name4), 0)
                        titles = [f"{name1}_{name2}_{name3}_{name4}_{i}" for i in range(count)]
                        for name in researcher_names:
                            venn_sets[name].update(titles)

                        fig = venny4py(sets=venn_sets)
                        st.pyplot(fig)

        # Record end time and calculate execution time
        execution_time = (time.time() - start_time)/60
        st.markdown(f"<h3 style='font-size: 30px;'>Simulation time: {execution_time:.2f} mins</h3>", unsafe_allow_html=True)

    else:
        st.warning("Please upload at least two Google Scholar PDFs to run the simulation.")


# import streamlit as st
# import fitz  
# from io import BytesIO
# import pandas as pd
# from itertools import combinations
# import matplotlib.pyplot as plt
# from venny4py.venny4py import venny4py  # Import venny4py for 4-set Venn diagrams
# import matplotlib_venn as venn  # Correct import for venn library
# from fuzzywuzzy import fuzz  # Import fuzzywuzzy for fuzzy string matching
# import time  # Import time module to track execution time
# import datetime
# import re  # For extracting years
# from upsetplot import UpSet, from_memberships

# # Disable the deprecation warning globally
# st.set_option('deprecation.showPyplotGlobalUse', False)

# # Set the threshold at the top of the code (you can change this value easily here)
# THRESHOLD = 95  # Set your default threshold here

# # Function to normalize publication titles
# def clean_title(title):
#     return " ".join(title.lower().strip().replace("-", " ").split())  # Normalize hyphens, spaces, and case

# # Strip the .pdf extension from file name
# def clean_filename(file_name):
#     return file_name.rsplit('.', 1)[0]  # Remove the .pdf extension

# def is_google_scholar(file_object):
#     if file_object is None or file_object.getbuffer().nbytes == 0:
#         file_name = getattr(file_object, "name", "Unknown file")
#         st.error(f"The uploaded file '{file_name}' is empty or invalid.")
#         return False

#     file_object.seek(0)
#     pdf_document = fitz.open(stream=file_object.read(), filetype="pdf")
#     page1 = pdf_document.load_page(0)

#     topmost_y = float('inf')
#     topmost_text = ""

#     for block in page1.get_text("dict")["blocks"]:
#         block_type = block.get("type", 0)
#         if block_type == 0: 
#             block_bbox = block.get("bbox", [])
#             if block_bbox and block_bbox[1] < topmost_y:
#                 topmost_y = block_bbox[1]
#                 topmost_text = " ".join(
#                     span['text'] for line in block.get("lines", [])
#                     for span in line.get("spans", [])
#                 )

#     pdf_document.close()
#     return "Google Scholar" in topmost_text

# def is_blue(color_int):
#     """Check if text color is blue (Google Scholar titles are blue)."""
#     r = (color_int >> 16) & 0xFF
#     g = (color_int >> 8) & 0xFF
#     b = color_int & 0xFF
#     return b > r and b > g

# def extract_titles_citations_years(file_object):
#     titles = []
#     citations = []
#     years = []
#     current_title = ""
#     found_first_title = False  

#     if file_object is None or file_object.getbuffer().nbytes == 0:
#         return []

#     file_object.seek(0)
#     pdf_document = fitz.open(stream=file_object.read(), filetype="pdf")

#     try:
#         for page_number in range(len(pdf_document)):
#             page = pdf_document.load_page(page_number)
#             text_lines = page.get_text("text").split("\n")  
#             blocks = page.get_text("dict")["blocks"]

#             for block in blocks:
#                 if "lines" in block:
#                     for line in block["lines"]:
#                         for span in line["spans"]:
#                             if is_blue(span['color']):  
#                                 current_title += span['text'] + " "  
#                                 found_first_title = True  
#                             else:
#                                 if current_title.strip():  
#                                     titles.append(current_title.strip())
#                                     current_title = ""  

#             for i in range(len(text_lines) - 1):  
#                 line = text_lines[i].strip()
                
#                 # Find a 4-digit year (must be after first title)
#                 if found_first_title and re.match(r"\b(20\d{2}|19\d{2})\b", text_lines[i + 1].strip()):
#                     year_match = re.search(r"\b(20\d{2}|19\d{2})\b", text_lines[i + 1].strip())  
#                     year = year_match.group(0) if year_match else "N/A"  
#                     years.append(year)

#                     if line.isdigit(): 
#                         citations.append(line)
#                     else:
#                         citations.append("N/A") 

#         max_length = max(len(titles), len(citations), len(years))
#         while len(titles) < max_length:
#             titles.append("N/A")
#         while len(citations) < max_length:
#             citations.append("N/A")
#         while len(years) < max_length:
#             years.append("N/A")

#     finally:
#         pdf_document.close()

#     extracted_data = list(zip(titles, years, citations))
#     return extracted_data  

# # Fuzzy string matching to find similar titles between two sets
# def find_similar_titles(set1, set2, threshold=THRESHOLD):  
#     matches = set()
#     for title1 in set1:
#         for title2 in set2:
#             if fuzz.ratio(title1, title2) >= threshold:  
#                 matches.add(title1)  
#     return matches

# # Add a button to run the simulation
# st.title("Google Scholar PubMatch")

# # Add the attribution text below the title
# st.markdown(
#     '"This common publications matching tool is implemented based on the work done by Rohit Ramachandran, Rutgers University, NJ, USA"'
# )

# # Add instructions banner
# st.markdown("""
# ### Instructions:
# 1. Upload at least 2 Google Scholar PDFs. Ensure all the papers are printed to PDF. See 'show more' at the end of the google scholar link.
# 2. The algorithm will check for combinations of common publications across 2 to N researchers, where N ~=8 before memory limitations hit. N=8 has 247 combinations.
# 3. Once the analysis is complete, download the comparison results as a CSV file.
# 4. You can also view the table, Upset plot and Venn diagram showing the overlap of publications between researchers. 
# 5. Venn diagrams are absent for 5 or more researchers. 
# """)

# # Get the current year dynamically
# current_year = datetime.datetime.now().year

# # Dropdown for year filtering
# year_options = ["All Years"] + list(range(current_year, 1950, -1))  
# selected_years = st.multiselect("Select years to include:", year_options, default=["All Years"])  

# # If "All Years" is selected, include all years
# if "All Years" in selected_years:
#     selected_years = set(map(str, range(current_year, 1950, -1)))  
# else:
#     selected_years = set(map(str, [year for year in selected_years if year != "All Years"]))  

# uploaded_files = st.file_uploader(
#     "Upload Google Scholar PDFs of researchers",
#     type=["pdf"],
#     accept_multiple_files=True
# )

# # Add a button for running the simulation
# run_simulation_button = st.button("Run Simulation")

# # If the button is pressed, start the simulation
# if run_simulation_button:
#     # Record start time
#     start_time = time.time()

#     if uploaded_files:
#         researcher_data = {}

#         for file in uploaded_files:
#             if is_google_scholar(file):
#                 researcher_name = clean_filename(file.name)  
#                 extracted_data = extract_titles_citations_years(file)
#                 # Filter titles by the selected years
#                 filtered_titles = { (title, year, citations) for title, year, citations in extracted_data if year in selected_years}
#                 researcher_data[researcher_name] = filtered_titles
#             # else:
#             #     st.warning(f"Skipping {file.name}: Not detected as a Google Scholar PDF.")

#         if len(researcher_data) > 1:
#             st.subheader("Common Publications Between Researchers")

#             all_sets = {name: set(t[0] for t in titles) for name, titles in researcher_data.items()}  # Only titles for comparison
#             all_data_with_details = {name: titles for name, titles in researcher_data.items()}  # Keep full data for citations
#             comparisons = []

#             # Generate all subset combinations (from pairs to all researchers)
#             for r in range(2, len(all_sets) + 1):  
#                 for comb in combinations(all_sets.keys(), r):
#                     common_titles = all_sets[comb[0]]
#                     for researcher in comb[1:]:
#                         common_titles = find_similar_titles(common_titles, all_sets[researcher])  
#                     comparisons.append({
#                         "Combinations": " ↔ ".join(comb),  
#                         "No. of common publications": len(common_titles),
#                         "Common Publications": ", ".join(common_titles),  # Changed to match venn_common_4.py (titles only)
#                     })

#             # Create the DataFrame with the common publications count
#             df_comparisons = pd.DataFrame(comparisons)

#             # Reset the index and start from 1
#             df_comparisons.index = range(1, len(df_comparisons) + 1)

#             # Display the table without "Common Publications"
#             st.dataframe(
#                 df_comparisons.drop(columns=["Common Publications"]).style.set_properties(**{
#                     'text-align': 'left',
#                     'white-space': 'pre-wrap'
#                 }),
#                 height=(50 + len(df_comparisons) * 35),  
#                 width=800
#             )

#             @st.cache_data
#             def convert_df(df):
#                 return df.to_csv(index=False).encode('utf-8')

#             # Calculate total common publications and citations for each researcher
#             researcher_totals = {name: {"publications": 0, "citations": 0} for name in researcher_data.keys()}
#             for _, row in df_comparisons.iterrows():
#                 researchers_in_comb = row["Combinations"].split(" ↔ ")
#                 common_titles = row["Common Publications"].split(", ")
#                 count = row["No. of common publications"]
                
#                 # Calculate citations for these common titles
#                 for researcher in researchers_in_comb:
#                     researcher_totals[researcher]["publications"] += count
#                     # Sum citations for common titles
#                     for title in common_titles:
#                         if title.strip():  # Ensure title is not empty
#                             for t, y, c in all_data_with_details[researcher]:
#                                 if fuzz.ratio(t, title.strip()) >= THRESHOLD and c != "N/A":
#                                     researcher_totals[researcher]["citations"] += int(c)

#             # Create a DataFrame for total common publications and citations
#             df_totals = pd.DataFrame(
#                 [(name, data["publications"], data["citations"]) for name, data in researcher_totals.items()],
#                 columns=["Researcher", "Total Common Publications", "Total Common Citations"]
#             )

#             # Reset the index to start from 1
#             df_totals.index = range(1, len(df_totals) + 1)

#             # Display the updated table
#             st.subheader("Total Common Publications and Citations per Researcher")
#             st.dataframe(
#                 df_totals.style.set_properties(**{
#                     'text-align': 'left',
#                     'white-space': 'pre-wrap'
#                 }),
#                 height=(50 + len(df_totals) * 35),
#                 width=800
#             )

#             # Provide option to download the table as CSV
#             csv_data = convert_df(df_comparisons)
#             st.download_button(
#                 label="Download Comparison as CSV",
#                 data=csv_data,
#                 file_name="publication_comparisons.csv",
#                 mime="text/csv"
#             )
            
#             # Generate the UpSet plot from the comparisons data dynamically
#             st.subheader("UpSet Plot of Common Publications")
#             researcher_names = list(researcher_data.keys())
#             upset_data = {}

#             for _, row in df_comparisons.iterrows():
#                 combination = row["Combinations"]
#                 common_publications_count = row["No. of common publications"]
#                 researchers_in_comb = combination.split(" ↔ ")
#                 membership = tuple(researcher in researchers_in_comb for researcher in researcher_names)
#                 if membership in upset_data:
#                     upset_data[membership] += common_publications_count
#                 else:
#                     upset_data[membership] = common_publications_count

#             index = pd.MultiIndex.from_tuples(upset_data.keys(), names=researcher_names)
#             data_counts = pd.Series(list(upset_data.values()), index=index)

#             upset = UpSet(data_counts, subset_size="auto", show_counts="%d", sort_by="cardinality", facecolor="blue")
#             fig = plt.figure(figsize=(12, 8))
#             upset.plot(fig=fig)
#             st.pyplot(fig)
                            
#             # Only plot the Venn diagram if there are 2, 3, or 4 researchers
#             if len(researcher_data) > 1:
#                 if len(researcher_data) > 4:
#                     st.warning("Venn diagrams are only available for up to 4 researchers. Skipping Venn diagram.")
#                 else:
#                     researcher_names = list(all_sets.keys())
#                     total_publications = {name: len(all_sets[name]) for name in researcher_names}

#                     intersection_counts = {}
#                     for r in range(1, len(researcher_names) + 1):
#                         for comb in combinations(researcher_names, r):
#                             intersection_counts[comb] = 0

#                     for _, row in df_comparisons.iterrows():
#                         researchers_in_comb = tuple(row["Combinations"].split(" ↔ "))
#                         count = row["No. of common publications"]
#                         intersection_counts[researchers_in_comb] = count

#                     singleton_counts = {}
#                     for name in researcher_names:
#                         total = total_publications[name]
#                         for comb, count in intersection_counts.items():
#                             if name in comb:
#                                 total -= count
#                         singleton_counts[name] = max(total, 0)

#                     if len(researcher_names) == 2:
#                         name1, name2 = researcher_names
#                         venn_data = [
#                             singleton_counts[name1],  
#                             singleton_counts[name2],  
#                             intersection_counts[(name1, name2)]  
#                         ]
#                         fig, ax = plt.subplots(figsize=(6, 6))
#                         v = venn.venn2(
#                             subsets=venn_data,
#                             set_labels=researcher_names
#                         )
#                         st.pyplot(fig)

#                     elif len(researcher_names) == 3:
#                         name1, name2, name3 = researcher_names
#                         venn_data = [
#                             singleton_counts[name1],  
#                             singleton_counts[name2],  
#                             intersection_counts[(name1, name2)],  
#                             singleton_counts[name3],  
#                             intersection_counts[(name1, name3)],  
#                             intersection_counts[(name2, name3)],  
#                             intersection_counts[(name1, name2, name3)]  
#                         ]
#                         fig, ax = plt.subplots(figsize=(6, 6))
#                         v = venn.venn3(
#                             subsets=venn_data,
#                             set_labels=researcher_names
#                         )
#                         st.pyplot(fig)

#                     elif len(researcher_names) == 4:
#                         name1, name2, name3, name4 = researcher_names
#                         venn_sets = {name: set() for name in researcher_names}
#                         for name in researcher_names:
#                             count = singleton_counts[name]
#                             venn_sets[name].update([f"{name}_unique_{i}" for i in range(count)])
                        
#                         for (n1, n2) in combinations(researcher_names, 2):
#                             count = intersection_counts[(n1, n2)]
#                             titles = [f"{n1}_{n2}_{i}" for i in range(count)]
#                             venn_sets[n1].update(titles)
#                             venn_sets[n2].update(titles)
                        
#                         for (n1, n2, n3) in combinations(researcher_names, 3):
#                             count = intersection_counts.get((n1, n2, n3), 0)
#                             titles = [f"{n1}_{n2}_{n3}_{i}" for i in range(count)]
#                             venn_sets[n1].update(titles)
#                             venn_sets[n2].update(titles)
#                             venn_sets[n3].update(titles)
                        
#                         count = intersection_counts.get((name1, name2, name3, name4), 0)
#                         titles = [f"{name1}_{name2}_{name3}_{name4}_{i}" for i in range(count)]
#                         for name in researcher_names:
#                             venn_sets[name].update(titles)

#                         fig = venny4py(sets=venn_sets)
#                         st.pyplot(fig)

#         # Record end time and calculate execution time
#         execution_time = (time.time() - start_time)/60
#         st.markdown(f"<h3 style='font-size: 30px;'>Simulation time: {execution_time:.2f} mins</h3>", unsafe_allow_html=True)

#     else:
#         st.warning("Please upload at least two Google Scholar PDFs to run the simulation.")
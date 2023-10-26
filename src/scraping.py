import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse, urljoin
from src.common_utils import get_embedding_with_retry,split_text
from src.common_utils import push_to_typesense



CREATE_COLLECTION = True

def scrap_main(starting_url,id, max_links=100):
    # Scrape internal links with a limit
    links = scrape_internal_links(starting_url, parent_domain=urlparse(starting_url).netloc, max_links=max_links)
    collection_name = id
    # Create a DataFrame from the results
    df = pd.DataFrame(links, columns=['url', 'level'])
    df = df.drop_duplicates(subset=['url'])
    df = scrape_data_from_dataframe(df)
    df['header'] = df['header'].astype(str).str.replace("\n"," ")
    df['content'] = df['content'].astype(str).str.replace("\n"," ")
    df = df.dropna()
    df['content'] = df['content'].astype(str)
    print(df.shape)

    # embedding
    # limit content

    # split
    # Split content into rows
    split_rows = []

    for _, row in df.iterrows():
        content = row['content']
        limit = 1500  # Adjust the character limit as needed
        rows = split_text(content, limit)
        for r in rows:
            split_rows.append({
                'url': row['url'],
                'title': row['title'],
                'header': row['header'],
                'content': r,
            })

    # Create a new DataFrame with the split rows
    result_df = pd.DataFrame(split_rows)
    print(result_df.shape)
    result_df = result_df.head(100)
    result_df['content_vector'] = result_df['content'].apply(lambda x: get_embedding_with_retry(x))

    push_to_typesense(result_df,collection_name)

    # Reset the index
    df = result_df.reset_index(drop=True)

    return df

# Function to scrape internal links with a limit
def scrape_internal_links(url, max_level=2, current_level=0, parent_domain=None, max_links=100):
    # Initialize a list to store the results
    links = []

    # Fetch the HTML content of the current URL
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract the domain of the current URL
        if parent_domain is None:
            parent_domain = urlparse(url).netloc

        # Extract all the anchor tags
        for a_tag in soup.find_all('a'):
            href = a_tag.get('href')
            if href:
                # Construct the absolute URL
                abs_url = urljoin(url, href)

                # Check if it's an internal link (i.e., same domain)
                parsed_url = urlparse(abs_url)
                if parsed_url.netloc == parent_domain:
                    links.append((abs_url, current_level))

                    # Check if the limit has been reached
                    if len(links) >= max_links:
                        return links  # Stop scraping when the limit is reached

                    # Recursively scrape internal links if the maximum level is not reached
                    if current_level < max_level:
                        links += scrape_internal_links(abs_url, max_level, current_level + 1, parent_domain, max_links)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")

    return links


# Function to scrape data from a URL
def scrape_url_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract the website title
            title = soup.title.string if soup.title else ""
            
            # Extract all header elements
            headers = ""
            header_tags = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            for header in header_tags:
                headers += header.text + "\n"
            
            # Extract and clean the content (remove HTML tags)
            content = ' '.join(soup.stripped_strings)
            
            return url, content, title, headers
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
    return url, None, None, None

# Main function to scrape data from URLs in the DataFrame
def scrape_data_from_dataframe(input_df):
    # Create a list to store the scraped data
    scraped_data = []

    for url in input_df['url']:
        result = scrape_url_data(url)
        scraped_data.append(result)

    # Create a new DataFrame from the scraped data
    output_df = pd.DataFrame(scraped_data, columns=['url', 'content', 'title', 'header'])

    return output_df
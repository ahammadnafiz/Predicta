import arxivscraper.arxivscraper as ax
import pandas as pd


scraper = ax.Scraper(category='stat',date_from='2024-01-01',date_until='2024-03-01',t=10, filters={'categories':['stat.ml'],'abstract':['learning']})
output = scraper.scrape()


cols = ('id', 'title', 'categories', 'abstract', 'doi', 'created', 'updated', 'authors')
df = pd.DataFrame(output,columns=cols)
csv_file_path = "arXivstat.csv"

df.to_csv(csv_file_path, index=False)

print("CSV file has been successfully created.")
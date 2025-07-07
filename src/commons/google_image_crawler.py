from icrawler.builtin import GoogleImageCrawler

# Create a crawler instance
google_crawler = GoogleImageCrawler(storage={'root_dir': 'lebron_images'})

google_crawler.crawl(
    keyword='lebron james playing',
    max_num=100,
    file_idx_offset=70
)

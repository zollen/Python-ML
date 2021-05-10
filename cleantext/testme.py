'''
Created on May 10, 2021

@author: zollen
@url: https://towardsdatascience.com/automate-your-text-processing-workflow-in-a-single-line-of-python-code-e276755e45de
@desc: CleanText is an open-source Python library that enables to clean of the textual 
        data scraped from the web or social media. CleanText enables developers to 
        create a normalized text representation. CleanText uses ftfy, unidecode, and 
        various other hard-coded rules including RegEx to convert a corrupted or dirty 
        input text into a clean text, that can be further processed to train an NLP model.
'''

from cleantext import clean
import warnings

warnings.filterwarnings('ignore')

s1 = 'Zürich'
print(clean(s1, fix_unicode=True))

s2 = "ko\u017eu\u0161\u010dek"
print(clean(s2, to_ascii=True))

s3 = "My Name is SATYAM"
print(clean(s3, lower=True))

s4 = "https://www.Google.com and https://www.Bing.com are popular seach engines. You can mail me at satkr7@gmail.com. If not replied call me at 9876543210"
print(clean(s4, no_urls=True, replace_with_url="URL", no_emails=True, 
      no_phone_numbers=True, replace_with_email="sample@gmail.com"))

s5 = "I want ₹ 40"
print(clean(s5, no_currency_symbols = True))
print(clean(s5, no_currency_symbols = True, replace_with_currency_symbol="Rupees"))

s7 = 'abc123def456ghi789zero0'
print(clean(s7, no_digits = True))
print(clean(s7, no_digits = True, replace_with_digit=""))

s6 = "40,000 is greater than 30,000."
print(clean(s6, no_punct = True))

text = """
Zürich has a famous website https://www.zuerich.com/ 
WHICH ACCEPTS 40,000 € and adding a random string, :
abc123def456ghi789zero0 for this demo. Also remove punctions ,. 
my phone number is 9876543210 and mail me at satkr7@gmail.com.' 
     """

clean_text = clean(text, 
      fix_unicode=True, 
      to_ascii=True, 
      lower=True, 
      no_line_breaks=True,
      no_urls=True, 
      no_numbers=True, 
      no_digits=True, 
      no_currency_symbols=True, 
      no_punct=True, 
      replace_with_punct="", 
      replace_with_url="<URL>", 
      replace_with_number="<NUMBER>", 
      replace_with_digit="", 
      replace_with_currency_symbol="<CUR>",
      lang='en')

print(clean_text)
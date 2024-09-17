# Web Scraping

## Basic Implementation

Use the bs4 python package for this.

### Imports

```python
from bs4 import BeautifulSoup # methods to get and parse data from a url
import requests # gets the page from a url
import lxml # parser class which beautiful soup integrates
```

### Methods

Get Page

```python
def get_page(self) -> BeautifulSoup:
        response = requests.get(self.url)
        if not response.ok:
            print("Server responded: ", response.status_code)
            return None
        else:
            soup = BeautifulSoup(response.text, "lxml")
            return soup
```

This method makes a request for page data given a url, then passes this response into the soup object initialization along with the specified parse "lxml" and returns the object.

Get Detailed Data

```python
def get_detailed_data(self, tag: str, class_: str = None) -> list:
        data = []
        try:
            for item in self.soup.find_all(tag, class_):
                data.append(item.text)
        except None:
            print("No data found")
        return data
```

This method takes in tag and class names for elements on the page the url points to. It then tries to find all occurences of these elements on the page and stores them into a list which is returned.

This method assumes you know you which tags and classess you are looking for already. You can find this information out by insepcting the page source manually.

import json

class data_parser:
    def __init__(self, category, business_file, review_file):
        self.category = set(category)
        self.businessfile = business_file
        self.reviewfile = review_file
        self.restaurant_id = []
        self.reviews = dict()

    def category_parser(self):
        # return get business_id of business related to restaurant and food
        f = open(self.businessfile,'r')
        lines = f.readlines()
        for line in lines:
            business = json.loads(line)
            category = business['categories']
            if not category == None:
                category = category.split()
                if not len(set(category).intersection(self.category)) == 0:
                    self.restaurant_id.append(business['business_id'])

    def review_parser(self):
        # return dictionary of {review:star}
        self.category_parser()
        f = open(self.reviewfile, 'r')
        lines = f.readlines()
        for line in lines:
            review = json.loads(line)
            if review['business_id'] in self.restaurant_id:
                self.reviews[review['text']] = review['stars']
        with open('reviews_label.txt', 'w') as outfile:
            json.dump(self.reviews, outfile)

if __name__ == '__main__':
    category = ['Restaurant', 'Food']
    business_file = 'business.json'
    review_file = 'review.json'
    data_parser = data_parser(category, business_file, review_file)
    data_parser.review_parser()


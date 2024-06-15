from posts_helpers import clean_input, get_stackoverflow_posts

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filename = "Architectural Posts.xlsx"
    new_filename = "cleaned architectural posts.xlsx"

    df = clean_input(filename, new_filename)

    posts = get_stackoverflow_posts(df['postId'].head(10))

    for post in posts:
        print(post)
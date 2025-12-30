import os
from dotenv import load_dotenv
load_dotenv()

from recommender import TVShowRecommender
import studio

def get_verified_titles(recommender):
    """
    Step 1 & 2: Interaction loop to get and verify user shows.
    """
    while True:
        print("\n" + "="*40)
        user_input = input("Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show:\n")
        
        raw_titles = [t.strip() for t in user_input.split(',') if t.strip()]
        
        if len(raw_titles) < 2:
            print("Please enter at least two shows.")
            continue
            
        verified_titles = []
        
        for raw in raw_titles:
            current_attempt = raw
            while True:
                matches = recommender.find_best_matches([current_attempt])
                best_match = matches[0]
                confirm = input(f"For '{current_attempt}', did you mean '{best_match}'? (y/n): ").lower().strip()
                
                if confirm == 'y':
                    verified_titles.append(best_match)
                    break 
                else:
                    print(f"Okay, let's fix '{current_attempt}'.")
                    current_attempt = input("Please write the correct name for this show: ").strip()
        
        return verified_titles

def main():
    try:
        # Initialize the "Brain"
        recommender = TVShowRecommender()
        
        # Step 1 & 2: Get Validated Input
        user_titles = get_verified_titles(recommender)
        
        # Step 3: Recommendations
        print("\nGreat! Generating recommendations now...")
        recommendations = recommender.recommend_shows(user_titles)
        
        # Step 4: Display Recommendations
        print("\nHere are the tv shows that I think you would love:")
        rec_titles_only = []
        for title, score in recommendations:
            print(f"{title} ({score}%)")
            rec_titles_only.append(title)
            
        # Step 5: Creative Studio (Generative AI)
        print("\n" + "="*40)
        print("I have also created just for you two shows which I think you would love.")
        
        studio.generate_fictional_show(user_titles, "the input shows that you gave me")
        studio.generate_fictional_show(rec_titles_only[:3], "the shows that I recommended for you")
        
        print("\nHere are also the 2 tv show ads. Hope you like them!")
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

if __name__ == "__main__":
    main()
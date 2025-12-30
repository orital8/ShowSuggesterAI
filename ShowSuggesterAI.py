from dotenv import load_dotenv
# Load environment variables FIRST, before other imports use them
load_dotenv()

from recommender import TVShowRecommender

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
        
        # Step 3: Transition
        print("\nGreat! Generating recommendations now...")
        
        # Step 4: Logic Implementation
        recommendations = recommender.recommend_shows(user_titles)
        
        print("\nHere are the tv shows that I think you would love:")
        for title, score in recommendations:
            print(f"{title} ({score}%)")
            
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise e

if __name__ == "__main__":
    main()
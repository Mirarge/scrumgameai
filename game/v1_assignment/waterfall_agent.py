class WaterfallAgent:
    # Waterfall agent for the Scrum Game environment.

    def choose_action(self, state):
        # Choose an action, which is always to continue the current sprint (action 0) if it is not completed,
        # otherwise switch to the next product (action 1).
        
        return 0  # Continue
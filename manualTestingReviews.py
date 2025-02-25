import torch

# File imports
import lexigraphicOrder

# Does not work, need to fix
def testing_input(hashing, model, padd_width):
    keep_going = True
    while keep_going:
        userReview = input("Give me a review: ")
        userReview = userReview.split()
        hashing2 = lexigraphicOrder.lex_order_new(hashing, userReview, padd_width)
        # Let's encode these strings as numbers using the dictionary from earlier
        padder = []
        lister = []
        for word in userReview:
            lister.append(hashing2[word])
        padder.append(torch.tensor(lister))

        testing_tensor = torch.nn.utils.rnn.pad_sequence(padder, batch_first=True)
        model.eval()

        if model(testing_tensor).item() < 0.5:
            print("Sounds like a negative reviewer is afoot")
        elif model(testing_tensor).item() >= 0.5:
            print("Somebody is brimming with positivity")
        else:
            print("Not to hot not to cold")
        print(model(testing_tensor))

        awnser = input("Do you want to try again?:  ")
        if awnser not in {"yes", "y"}:
            print("ARE YOU DOUBLE SURE YOU'RE DONE?")
            awnser = input()

            if awnser not in {"yes", "y"}:
                print("ok you are")
                keep_going = False

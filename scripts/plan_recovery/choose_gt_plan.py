# Author: Ma'ayan Armony <maayan.armony@kcl.ac.uk>
# Class to choost the ground truth plan against which the generated plan should be evaluated.
# This is done by comparing the generated plan to all possible ground truth plans and choosing the one with the smallest
# cost, and the largest subsequence

from common import get_largest_subset, get_largest_subsequence, check_special_states


class ChooseGTPlan:
    # TODO change so that if the GT plan is a subset of the current plan, it is chosen?
    """
    Favourable:
    1. Same plan
    2. Plan with largest subsequence if includes init and goal states
    3. Plan with largest subset (consecutive actions)
    4. Plan with largest subsequence if much longer than largest subset
    5. Plan with smallest cost
    """
    def __init__(self, generated_plan: list[str], gt_plans: list[list[str]]):
        self.gen_plan = generated_plan
        self.gt_plans = gt_plans
        self.gt_plan = []
        self.generation_required = False
        self.score = 0

    def get_score(self) -> float:
        """
        Get the score of the generated plan against the chosen ground truth plan
        :return: the score of the generated plan against the chosen ground truth plan
        """
        return self.score

    def get_gt_plan(self) -> list[str]:
        """
        Get the chosen ground truth plan against which the generated plan was evaluated
        :return: the chosen ground truth plan
        """
        return self.gt_plan

    def choose_gt_plan(self) -> None:
        """
        Choose the ground truth plan against which to evaluate the generated plan, and set the class's gt_plan and
        score attributes accordingly.
        """
        # TODO need to check criteria here as well (e.g. init and goal scores?)
        plan_scores = self.get_plan_scores()
        best_plan = max(plan_scores, key=plan_scores.get)  # get the plan with the highest score (tuple)

        self.score = plan_scores[best_plan]
        self.gt_plan = list(best_plan)
        # return best_plan

    def get_plan_scores(self) -> dict:
        """
        Get the scores of the generated plan against all ground truth plans
        :return: a dictionary of the scores of the generated plan against all ground truth plans
        """
        plan_scores = {}

        for plan in self.gt_plans:
            self.generation_required = False
            score = self.compute_plan_score(self.gen_plan, plan)
            plan_scores[tuple(plan)] = score

        return plan_scores

    def compute_plan_score(self, curr_plan, gt_plan) -> float:
        """
        Compute the similarity score of the generated plan against the ground truth plan
        :param curr_plan: the generated plan to evaluate
        :param gt_plan: the ground truth plan to compare against
        :return: a score of similarity between the two plans
        """
        # 1. Same plan
        if curr_plan == gt_plan:
            score = max(10.0, len(curr_plan) * 2)  # 2 * len(plan) is same as 2 * longest subseq
            self.gt_plan = gt_plan  # set the GT plan as the current plan because they are the same
            # print(f"Same plan, score: {score}")
        else:
            init, goal = check_special_states(curr_plan, gt_plan)
            subseq = get_largest_subsequence(curr_plan, gt_plan)
            subset = get_largest_subset(curr_plan, gt_plan)

            # 2. Plan with the largest subsequence if includes init and goal states
            if init + goal == 2:
                score = 2 * len(subseq) + 1
                # print(f"Init and goal, subseq length {len(subseq)}, score: {score}")
            elif init + goal > 0:
                score = len(subseq) + len(subset) + 1  # subseq >= subset for any plan
                # print(f"subsequence: {subseq}, subset: {subset}")
                # print(f"Init or goal, subseq length {len(subseq)}, subset length {len(subset)}, score: {score}")
            else:
                # 4. Plan with the largest subsequence if much longer than the largest subset
                if len(subseq) > len(subset) * 1.5:
                    score = 1.5 * len(subseq)   # TODO check if this makes sense
                    # print(f"Subseq much longer, score: {score}")
                # 3. Plan with the largest subset (consecutive actions)
                else:
                    score = 2 * len(subset)
                    # print(f"Subset, score: {score}")

            # If generation is definitely required, reduce score (regardless of length, this is dealt with later)
            if self.generation_required:
                score -= 1

            # 5. Plan with the smallest cost (penalise on difference in length)  # TODO add cost
            # Linearly penalise difference in length
            if len(curr_plan) != 0:
                score += self.penalise_length_diff(curr_plan, gt_plan) / len(curr_plan)
            else:
                score += self.penalise_length_diff(curr_plan, gt_plan)
        return score

    def penalise_length_diff(self, curr_plan, gt_plan) -> float:
        """
        Reduce the score of plans depending on their difference from the ground truth plan.
        Penalises shorter plans more - longer plans can be suboptimal, shorter plans are necessarily missing vital actions.
        """
        # TODO if all GT plans are longer, penalise more? Can be done automatically if normalising with length of plan?

        length_diff = len(curr_plan) - len(gt_plan)

        if length_diff > 0:
            score = -1 * length_diff # current plan may include the GT plan
        elif length_diff == 0:
            score = 0 # no penalty if plans are the same length
        else:
            score = 2 * length_diff # current plan may be subset of GT plan, a new plan will definitely need to be generated
            self.generation_required = True

        # print(f"Length diff: {length_diff}, score: {score}")
        return score

    def get_plan_cost(self, plan) -> float:
        """
        Get the cost of the plan
        :param plan: the plan to evaluate
        :return: the cost of the plan
        """
        pass


def main():
    chooser = ChooseGTPlan(["a", "b", "c", "d"], [["a", "b", "c"], ["a", "b", "c", "e"], ["a", "b", "c", "d", "e", "f"]])
    chooser.choose_gt_plan()
    print(f"Chosen plan: {chooser.get_gt_plan()}")

if __name__ == "__main__":
    main()
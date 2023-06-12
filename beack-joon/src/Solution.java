import java.util.*;

class Solution {

    public static void main(String[] args) {
        int[] arr = {0,1,2,3,4,5};
        int[] query = {4,1,2};
        System.out.println(solution(arr,query));
    }
    public static ArrayList<Integer> solution(int[] arr, int[] query) {
        ArrayList<Integer> answer = new ArrayList(Arrays.asList(arr));

        for(int i = 0;  i < query.length; i++) {
            if(i % 2 == 0){
                int k = answer.size();
                for(int j = query[i]+1; j < k; j++)
                    answer.remove(answer.size()-1);
            }
            else {
                for(int j = 0; j < query[i]; j++)
                    answer.remove(0);
            }
        }
        return answer;
    }
}
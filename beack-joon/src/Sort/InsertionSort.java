package Sort;

import java.util.*;

//Insertion Sort
//////////////////////////////////
// 2번째 원소부터 시작하여 그 앞의 원소들과 비교하여 삽입 위치를 지정 한 후,
// 원소를 뒤로 옮기고 지정된 자리에 자료를 삽입
////////////////////////////////
//과정
// 1.두번째 위치의 값을 temp에 저장
// 2. temp와 이전의 원소들과 비교하며 삽입
// 3. 1로 돌아가 다음 값을 temp에 저장 후, 반복
//////////////////////////////////
// 시간 복잡도 : 최선 : O(n), 평균/최악 : O(n^2)
//////////////////////////////////
public class InsertionSort {
    void insertionSort(int[] arr)
    {
        for(int index = 1 ; index < arr.length ; index++){ // 1.
            int temp = arr[index];
            int prev = index - 1;
            while( (prev >= 0) && (arr[prev] > temp) ) {    // 2.
                arr[prev+1] = arr[prev];
                prev--;
            }
            arr[prev + 1] = temp;                           // 3.
        }
        System.out.println(Arrays.toString(arr));
    }
}

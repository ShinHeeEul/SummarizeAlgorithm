package Sort;
import java.util.*;

//Selection Sort
//////////////////////////////////
// 해당 순서에 원소를 넣을 위치는 이미 정해져 있고, 어떤 원소를 넣을 지 선택하는 알고리즘
// 해당 자리를 선택하고 그 자리의 값을 찾아오는 것
//////////////////////////////////
//과정
// 1. 주어진 배열중 최소/최대값 찾음
// 2. 그 값을 맨 앞에 값과 교체
// 3. 맨 처음 위치 제외후 반복
//////////////////////////////////
// 시간 복잡도 : 평균/최선/최악 O(n^2)
//////////////////////////////////
public class SelectionSort {

    void selectionSort(int[] arr) {
        int indexMin, temp;
        for (int i = 0; i < arr.length-1; i++) {        // 1.
            indexMin = i;
            for (int j = i + 1; j < arr.length; j++) {  // 2.
                if (arr[j] < arr[indexMin]) {           // 3.
                    indexMin = j;
                }
            }
            // 4. swap(arr[indexMin], arr[i])
            temp = arr[indexMin];
            arr[indexMin] = arr[i];
            arr[i] = temp;
        }
        System.out.println(Arrays.toString(arr));
    }

}

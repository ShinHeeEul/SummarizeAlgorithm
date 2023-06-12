package Sort;

import java.util.*;

//BubbleSort
//////////////////////////////////
//서로 인접한 두 우너소의 대소를 비교하고, 자리를 교환하며 정렬하는 알고리즘
//원소의 이동이 마치 거품이 수면위로 올라오는 것 같다해서 버블소트
//////////////////////////////////
//과정
// 1. 1회전에 첫번째vs두번째, 두번째vs 세번째.. 마지막까지 교환
// 2. sort 범위를 하나씩 줄이고 1 반복
//////////////////////////////////
// 시간 복잡도 : 평균/최선/최악 O(n^2)
//////////////////////////////////

public class BubbleSort {

    void bubbleSort(int[] arr) {
        int temp = 0;
        for(int i = 0; i < arr.length; i++) {       // 1.
            for(int j= 1 ; j < arr.length-i; j++) { // 2.
                if(arr[j-1] > arr[j]) {             // 3.
                    // swap(arr[j-1], arr[j])
                    temp = arr[j-1];
                    arr[j-1] = arr[j];
                    arr[j] = temp;
                }
            }
        }
        System.out.println(Arrays.toString(arr));
    }
}

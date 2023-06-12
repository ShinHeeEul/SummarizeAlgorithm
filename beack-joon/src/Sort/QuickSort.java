package Sort;

//Heap Sort
//////////////////////////////////
// 분할 정복 방법을 통해 주어진 배열을 정렬
// 불안정 정렬
// pivot을 이용
//////////////////////////////////
//과정
// 1. 배열 가운데서 하나의 원소를 고름(pivot)
// 2. 피벗 앞에는 피벗보다 값이 작은 모든 원소가 오게,
//뒤에는 피벗보다 값이 큰 모든 원소가 오게 배열을 둘로 나눔
// --> 이를 분할이라 함
// 3. 분할된 두 개의 작은 배열에 대해 재귀적으로 이 과정을 반복

// 재귀 호출이 한번 진행될 때마다 최소 하나의 원소는 최종적으로 위치가 정해지므로,
// 이 알고리즘은 반드시 끝남
//////////////////////////////////
// 시간 복잡도 : 최선/평균 : O(nlogn), 최악 : O(n^2)
//////////////////////////////////
//Java의 Arrays.sort 내부적으로 Dual Pivot Quick Sort로 구현
public class QuickSort {

    //정복
    public void quickSort(int[] array, int left, int right) {
        if(left >= right) return;

        // 분할
        int pivot = partition(array, left, right);

        // 피벗은 제외한 2개의 부분 배열을 대상으로 순환 호출
        quickSort(array, left, pivot-1);  // 정복(Conquer)
        quickSort(array, pivot+1, right); // 정복(Conquer)
    }

    //분할
    public int partition(int[] array, int left, int right) {
        /**
         // 최악의 경우, 개선 방법
         int mid = (left + right) / 2;
         swap(array, left, mid);
         */

        int pivot = array[left]; // 가장 왼쪽값을 피벗으로 설정
        int i = left, j = right;


        while(i < j) {
            //우측 값이 pivot보다 크면 놔두고 다음 값 검사
            while(pivot < array[j]) {
                j--;
            }
            //좌측 값이 pivot보다 작으면 다음 값 검사
            while(i < j && pivot >= array[i]){
                i++;
            }
            //여기까지 도달했다면 i와 j가 둘다 pivot 기준 반대라는 것으로 서로 바꿔줌
            swap(array, i, j);
        }
        //partition이 종료되면 pivot 값을 중간에 삽입해줌
        array[left] = array[i];
        array[i] = pivot;

        return i;
    }

    public static void swap(int[] array, int a, int b) {
        int temp = array[a];
        array[a] = array[b];
        array[b] = temp;
    }
}

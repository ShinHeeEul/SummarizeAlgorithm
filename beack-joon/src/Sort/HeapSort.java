package Sort;


//Heap Sort
//////////////////////////////////
//불안정 정렬
//완전 이진 트리(삽입할 때 왼쪽부터 차례로 추가하는 이진 트리) 기반 정렬
//////////////////////////////////
//과정
// 1. 최대 힙을 구성
// 2. 현재 힙 루트인 가장 큰 값을 마지막 요소와 바꾼 후, 힙 사이즈 하나 줄임
// 3. 힘의 사이즈가 1보다 크면 위 과정 반복
//////////////////////////////////
// 시간 복잡도 : 평균/최선/최악 O(nlogn)
//////////////////////////////////
//힙소트 활용도
// - 가장 크거나 가장 작은 값을 구할 때
// - 최대 k만큼 떨어진 요소들을 정렬할 때
public class HeapSort {

    public HeapSort() {
    }


    //main
    private void solve() {
        int[] array = { 230, 10, 60, 550, 40, 220, 20 };

        heapSort(array);

        for (int v : array) {
            System.out.println(v);
        }
    }


    //heapify Method
    // 최대 힙으로 정렬
    // 가장 큰 값을 루트로 보내고, 자기 자신도 위치를 찾아가는..? 그런 느낌
    //Parameter : array[] -> 트리, n -> 노드의 갯수이자 마지막 노드, i -> 현재 노드의 위치..?
    public static void heapify(int array[], int n, int i) {
        int p = i;
        int l = i * 2 + 1;
        int r = i * 2 + 2;

        //왼쪽 노드와 오른쪽 노드 중 더 큰 값으로 현재 값 초기화
        if (l < n && array[p] < array[l]) {
            p = l;
        }
        if (r < n && array[p] < array[r]) {
            p = r;
        }

        // 정렬할 게 더 이상 없으면 종료
        if (i != p) {
            swap(array, p, i);
            heapify(array, n, p);
        }
    }

    //heapSort method
    public static void heapSort(int[] array) {
        int n = array.length;

        // init, max heap
        //max heap으로 완전 이진 트리 만듦
        // i = n/2 - 1 -> 0 인 이유는 부모 노드의 인덱스를 기준으로 왼쪽 자식 노드는 2 * i + 1, 오른쪽 자식 노드는 2 * i + 2이기 때문
        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(array, n, i);
        }

        // for extract max element from heap
        // 트리의 값이 0이 될때까지 과정 반복
        //마지막 노드에 가장 큰 값을 배치 시키네 그리고 1 줄이고.
        for (int i = n - 1; i > 0; i--) {
            //루트와 마지막 노드 바꾸고 사이즈 1 줄이고
            swap(array, 0, i);
            // 최대 힙으로 정렬하고
            heapify(array, i, 0);
        }
    }

    //swap Method
    // 두 노드의 위치를 바꿈
    public static void swap(int[] array, int a, int b) {
        int temp = array[a];
        array[a] = array[b];
        array[b] = temp;
    }
}

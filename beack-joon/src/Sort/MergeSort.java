package Sort;

import java.util.*;

//Merge Sort
public class MergeSort {

    public void mergeSort(int[] array, int left, int right) {

        if(left < right) {
            int mid = (left + right) / 2;

            mergeSort(array, left, mid);
            mergeSort(array, mid+1, right);
            merge(array, left, mid, right);
        }

    }

    public static void merge(int[] array, int left, int mid, int right) {
        int[] L = Arrays.copyOfRange(array, left, mid + 1);
        int[] R = Arrays.copyOfRange(array, mid + 1, right + 1);

        int i = 0, j = 0, k = left;
        int ll = L.length, rl = R.length;

        while(i < ll && j < rl) {
            if(L[i] <= R[j]) {
                array[k] = L[i++];
            }
            else {
                array[k] = R[j++];
            }
            k++;
        }

        // remain
        while(i < ll) {
            array[k++] = L[i++];
        }
        while(j < rl) {
            array[k++] = R[j++];
        }
    }
}

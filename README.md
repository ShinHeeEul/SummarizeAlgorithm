# 알고리즘 공부 정리


# 안정 정렬 vs 불안정 정렬


## 안정 정렬 (Stable Sort)
- 안정 정렬은 중복된 값을 입력 순서와 동일하게 정렬하는 정렬 알고리즘
- ex) 삽입 정렬, 병합 정렬, 버블 정렬

## 불안정 정렬 (Unstable Sort)
- 중복된 값이 입력 순서와 동일하지 않게 정렬되는 알고리즘
- ex) 퀵 정렬, 선택 정렬, 계수 정렬


# 정렬 종류

## 버블 정렬 (Bubble Sort)

## 계수 정렬 (Counting Sort)

## 힙 정렬 (Heap Sort)

## 삽입 정렬 (Insertion Sort)

## 합병 정렬 (Merge Sort)

## 퀵 정렬 (Quick Sort)

## 기수 정렬 (Radix Sort)

## 선택 정렬 (Selection Sort)

## 위상 정렬(Topological Sort)
- 선후 관계가 정의된 그래프 구조에서 정렬을 하기 위해 사용
- 끝난 후 진입차수가 0이 아닌 노드가 있다면 순환이 존재한다는 것
- 시간 복잡도 : O(V + E)

1) 그래프의 각 노드들의 진입 차수 테이블 생성 및 진입 차수 계산
2) 진입 차수가 0인 노드를 큐에 넣기
3) 큐에서 노드를 꺼내고 꺼낸 노드 출력
4) 꺼낸 노드와 간선으로 연결된 노드들의 진입 차수--
5) 진입 차수가 0인 노드가 있다면 큐에 넣기
6) 3~5를 큐에 더 이상 아무것도 없을 때까지 반복

- 코드 템플릿

```java
class Topological {
    public void topological() {
        
    }
}
```

# DFS vs BFS
- 너무 쉽다 이건


# 이분탐색
- 시간복잡도 : O(logN)

# 유니온-파인드(Union Find)
- 그래프 알고리즘으로 두 노드가 같은 그래프에 속하는지 판별하는 알고리즘.
- 서로소 집합, 상호 베타적 집합, 분리 집합으로도 불림
- union(x,y) 연산과 find(x) 연산으로 이루어져 있다.
- union(x,y) 연산은 x와 y의 그래프를 합친다는 의미이며, find(x)는 x가 어느 그래프에 속하는지 연산한다는 의미이다.
- 일반적으로 O(logN), 최대 O(N), 경로 압축시 O(a(N)). a(N)은 아커만 함수를 의미 : 상수 시간 복잡도로 생각해도 됨

- union(x,y)
  1) x와 y의 부모노드를 찾는다.
  2) 둘이 같은 그래프라면 false를 반환하고
  3) x가 y보다 작다면 arr[y] = x, 반대라면 arr[x] = y

- find(x)
  1) parent[x] == x라면 x를 반환하고
  2) 아니라면 find(parent[x]);
     - 이 과정에서 parent[x] = find(parent[x])를 반환하면 경로 압축을 할 수 있다.

- 코드 템플릿
```java
class UnionFind {
    boolean union(int x, int y) {
        x = find(x);
        y = find(y);
        
        if(x == y) {
            return false;
        }
        if(x < y) {
            arr[x] = y;
        } else {
            arr[y] = x;
        }
        
        return true;
    }
    
    int find(int x) {
        if(x == arr[x]) {
            return arr[x];
        }
        return arr[x] = find(arr[x]);
    }
}
```

# 그래프 최단 거리 구하는 알고리즘들
## 밸만-포드 알고리즘
- 그래프 최단 경로 구하는 알고리즘
- 하나의 정점에서 출발하는 최단 거리를 구하며 음수 사이클이 없어야 함.(음수 가중치는 허용)
- 다익스트라에 비해 조금 더 시간복잡도가 증가한 대신 음수 가중치를 처리할 수 있음.
- 시간복잡도 : O(EV)

1) 출발 노드는 0 나머지는 INF로 초기화
2) 간선 m개를 하나씩 살펴보며 dist[v]가 무한대가 아닐 때 dist[v] = min(dist[v], dist[w] + cost(w,v))
3) 1)~2)과정을 모든 노드간 반복

- 음수 사이클 확인하기
  - n개 만큼 반복하는 과정을 한번 더 했을 때 바뀌는 값이 있다면 음수 사이클이 존재하는 것.

- 코드 템플릿
```java
class BellmanFord {
    class Edge {
        int v; //나가는 노드
        int w; //들어오는 노드
        int cost;
        
        public Edge(int v, int w, int cost) {
            this.v = v;
            this.w = w;
            this.cost = cost;
        }
    }
    
    static ArrayList<Edge> graph;
    
    public boolean BellmanFord(int n, int m, int start) {
        int[] dist = new int[n + 1];
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[start] = 0;
        
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                Edge edge = graph.get(j);
                if(dist[edge.v] != Integer.MAX_VALUE) {
                    dist[edge.w] = Math.min(dist[edge.w], dist[edge.v] + edge.cost);
                }
            }
        }

        //음수 사이클 검증
        for(int j = 0; j < m; j++) {
            Edge edge = list.get(j);
            if(dist[edge.v] != Integer.MAX_VALUE && dist[edge.w] > dist[edge.v] + edge.cost) {
                System.out.println("음수");
                return;
            }

        }
    }
}
```
## 플로이드-워셜 알고리즘

## 다익스트라 알고리즘
- 음의 가중치가 없는 그래프의 한 노드에서 각 모든 노드까지의 최단거리를 구하는 알고리즘을 말함.
- 기본적으로 그리디 + 다이나믹 프로그래밍 기법
- 시간복잡도 : O(ElogV)

1) 아직 방문하지 않은 정점 중 출발지로부터 가장 거리가 짧은 정점을 방문
2) 해당 정점을 거쳐 갈 수 있는 정점의 거리가 이전 기록한 값보다 적으면 갱신

- 코드 템플릿

```java
import java.util.PriorityQueue;

class Dijkstra {
    class Node implements Comparable<Node> {
        int index;
        int cost;

        public Node(int index, int cost) {
            this.index = index;
            this.cost = cost;
        }

        @Override
        public int compareTo(Node o) {
            return Integer.compare(this.cost, o.cost);
        }
    }

    static ArrayList<Node>[] graph;

    public void Dijkstra(int n, int start) {
        boolean[] check = new boolean[n + 1];
        int[] dist = new int[n + 1];
        final int INF = Integer.MAX_VALUE;
        Arrays.fill(dist, INF);
        dist[start] = 0;

        PriorityQueue<Node> pq = new PriorityQueue<>();
        pq.offer(new Node(start, 0));
        
        while(!pq.isEmpty()) {
            int nowVertex = pq.poll().index;
            
            if(check[nowVertex]) continue;
            check[nowVertex] = true;
            
            for(Node next : graph[nowVertex]) {
                if(dist[next.index] > dist[nowVertex] + next.cost) {
                    dist[next.index] = Math.min(dist[next.index], dist[nowVertex] + next.cost);
                    pq.offer(new Node(next.index, dist[next.index]));
                }
            }
        }
    }
}
```


# LCS(Longest Common Subsequence) : 최장 부분 수열 알고리즘
- dp를 응용한 최장 부분 수열 알고리즘
- 시간복잡도 : O(nm)

1) dp를 2차원 배열(arr)로 선언
2) 문자를 하나씩 비교
   1) 일치하는 문자가 나오면 arr[i][j] = arr[i-1][j-1] + 1
   2) 일치하지 않는 문자가 나온다면 arr[i][j] = max(arr[i-1][j], arr[i][j-1])
3) 배열이 끝날때까지 반복

- LCS 값을 구하는 경우 이를 역추적
1) arr[i-1][j] or arr[i][j-1] 중 같은 값이 있다면 거기로 이동
2) 없다면 arr[i-1][j-1]로 이동하면서 arr[i][j] 값을 answer 문자열 앞에 더하기
3) 반복이 끝나면 answer 출력

# LIS(Longest Increasing Subsequence) : 최장 증가 부분 수열
- 어떠한 수열에서 오름차순으로 증가하는 가장 긴 부분수열을 찾는 것

## dp를 이용한 방법
   - 정확한 LIS를 구할 때 사용
   - 시간 복잡도 : O(N ^ 2)
   1) k의 앞 순서에 있는 모든 원소들 중 값이 k보다 작은 원소에 대해, 각각의 원소에서 끝나는 최장 증가 수열의 길이를 알고 있으면 구할 수 있음.
   2) dp[i] = i번째 인덱스에서 끝나는 최장 증가 수열의 길이

## 이분 탐색을 이용한 방법
   - 정확한 LIS가 아닌 길이만 구할 때 사용
   - 시간 복잡도 : O(NlogN)
   1) 배열을 하나 더 두고 원래 수열에서의 각 원소에 대해 LIS 배열 내에서의 위치를 찾음.
   2) 기존 배열을 탐색 
   2) 현재 수가 배열의 마지막 값보다 크다면 LIS 내 배열에 추가
   3) 현재 수가 배열의 마지막 값보다 작다면 이분 탐색을 통해 들어갈 위치를 찾고 교체.
      1) 이분 탐색 시, start = 0, end = 현재 위치이며 start < end동안 반복 -> end 값으로 교체

# 투 포인터

# MST(Minimum Spanning Tree)

# Bit Masking